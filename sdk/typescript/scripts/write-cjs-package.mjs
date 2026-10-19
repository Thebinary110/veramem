// Mark dist/cjs as CommonJS so Node does not treat it as ESM (the root package is "type": "module").
import { writeFileSync } from "node:fs";
writeFileSync(new URL("../dist/cjs/package.json", import.meta.url), '{ "type": "commonjs" }\n');
