#!/usr/bin/env bash
# Apply VeraMem's repository security settings with the GitHub CLI.
#
#   gh auth login            # as a repository admin
#   scripts/harden-github.sh OWNER/REPO
#
# Idempotent: safe to re-run. Every setting here is also described in GOVERNANCE.md
# so it can be audited or applied by hand.
set -euo pipefail

REPO="${1:?usage: harden-github.sh OWNER/REPO}"
OWNER="${REPO%%/*}"
ADMIN_ID="$(gh api "users/${OWNER}" --jq .id)"
api() { gh api -H "Accept: application/vnd.github+json" "$@"; }

echo "==> Repository options (squash-only, auto-delete branches, no wiki)"
api -X PATCH "repos/${REPO}" \
  -F allow_merge_commit=false -F allow_rebase_merge=false -F allow_squash_merge=true \
  -F delete_branch_on_merge=true -F has_wiki=false -F allow_auto_merge=false \
  -F web_commit_signoff_required=true >/dev/null

echo "==> Secret scanning, push protection, Dependabot"
api -X PATCH "repos/${REPO}" --input - >/dev/null <<'JSON' || echo "   (secret scanning needs a public repo or GHAS)"
{"security_and_analysis": {
  "secret_scanning": {"status": "enabled"},
  "secret_scanning_push_protection": {"status": "enabled"}}}
JSON
api -X PUT "repos/${REPO}/vulnerability-alerts" >/dev/null
api -X PUT "repos/${REPO}/automated-security-fixes" >/dev/null
api -X PUT "repos/${REPO}/private-vulnerability-reporting" >/dev/null

echo "==> Actions: read-only token, no PR approvals by bots, SHA-pinned allow-list"
api -X PUT "repos/${REPO}/actions/permissions/workflow" \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false >/dev/null
api -X PUT "repos/${REPO}/actions/permissions" \
  -F enabled=true -f allowed_actions=selected -F sha_pinning_required=true >/dev/null \
  || api -X PUT "repos/${REPO}/actions/permissions" -F enabled=true -f allowed_actions=selected >/dev/null
api -X PUT "repos/${REPO}/actions/permissions/selected-actions" --input - >/dev/null <<'JSON'
{"github_owned_allowed": true, "verified_allowed": false,
 "patterns_allowed": ["astral-sh/setup-uv@*", "step-security/harden-runner@*",
   "ossf/scorecard-action@*", "pypa/gh-action-pypi-publish@*",
   "trufflesecurity/trufflehog@*", "sigstore/gh-action-sigstore-python@*"]}
JSON
echo "==> Workflows from outside contributors always need maintainer approval"
api -X PUT "repos/${REPO}/actions/permissions/fork-pr-contributor-approval" \
  -f approval_policy=all_external_contributors >/dev/null \
  || echo "   (set manually: Settings > Actions > 'Require approval for all external contributors')"

echo "==> Ruleset: protect main"
# Admins may merge their own pull requests without a second reviewer (a solo maintainer
# cannot approve their own PR), but only through a PR: direct pushes stay blocked.
existing="$(api "repos/${REPO}/rulesets" --jq '.[] | select(.name=="protect-main") | .id' || true)"
method=POST; path="repos/${REPO}/rulesets"
if [ -n "${existing}" ]; then method=PUT; path="repos/${REPO}/rulesets/${existing}"; fi
api -X "${method}" "${path}" --input - >/dev/null <<JSON
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request"}],
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "required_linear_history"},
    {"type": "required_signatures"},
    {"type": "pull_request", "parameters": {
      "required_approving_review_count": 1,
      "dismiss_stale_reviews_on_push": true,
      "require_code_owner_review": true,
      "require_last_push_approval": true,
      "required_review_thread_resolution": true,
      "allowed_merge_methods": ["squash"]}},
    {"type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": true,
      "required_status_checks": [
        {"context": "lint"}, {"context": "build"},
        {"context": "test (ubuntu-latest, 3.12)"},
        {"context": "codeql"}, {"context": "secrets"}, {"context": "audit"},
        {"context": "gate"}]}}
  ]
}
JSON

echo "==> Ruleset: only admins may create or move release tags"
existing="$(api "repos/${REPO}/rulesets" --jq '.[] | select(.name=="protect-tags") | .id' || true)"
method=POST; path="repos/${REPO}/rulesets"
if [ -n "${existing}" ]; then method=PUT; path="repos/${REPO}/rulesets/${existing}"; fi
api -X "${method}" "${path}" --input - >/dev/null <<JSON
{
  "name": "protect-tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
  "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"},
            {"type": "required_signatures"}]
}
JSON

echo "==> Environment 'pypi': maintainer approval, release tags only"
api -X PUT "repos/${REPO}/environments/pypi" --input - >/dev/null <<JSON
{"wait_timer": 0, "prevent_self_review": false,
 "reviewers": [{"type": "User", "id": ${ADMIN_ID}}],
 "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
api -X POST "repos/${REPO}/environments/pypi/deployment-branch-policies" \
  -f name='v*' -f type=tag >/dev/null 2>&1 || true

echo "==> Labels"
api -X POST "repos/${REPO}/labels" -f name=needs-vouch -f color=d93f0b \
  -f description="Author is not yet a vouched contributor" >/dev/null 2>&1 || true

cat <<EOF

Done. Remaining one-time steps (cannot be automated through the API):
  1. On https://pypi.org/manage/account/publishing/ add a trusted publisher:
       project veramem, owner ${OWNER}, repo ${REPO#*/}, workflow release.yml, environment pypi
  2. Turn on 2FA (hardware key recommended) for every maintainer account.
  3. Set up commit signing locally (git config commit.gpgsign true) since main requires signatures.
EOF
