import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const read = (path) => readFileSync(resolve(root, path), 'utf8')

const requiredFiles = [
  'package.json',
  'index.html',
  'vite.config.ts',
  '.env.example',
  'src/App.tsx',
  'src/styles.css',
  'src/types.ts',
  'src/lib/api.ts',
  'src/components/UploadPanel.tsx',
  'src/components/JobCard.tsx',
  'src/components/JobDetail.tsx',
  'src/components/ScoreRing.tsx',
]

const app = read('src/App.tsx')
const api = read('src/lib/api.ts')
const css = read('src/styles.css')
const upload = read('src/components/UploadPanel.tsx')
const detail = read('src/components/JobDetail.tsx')
const packageJson = JSON.parse(read('package.json'))

const checks = {
  required_files_present: requiredFiles.every((file) => existsSync(resolve(root, file))),
  react_typescript_vite: Boolean(packageJson.dependencies?.react && packageJson.devDependencies?.typescript && packageJson.devDependencies?.vite),
  production_api_endpoint: api.includes('/analyze-resume?top_k=') && api.includes("'/model'"),
  environment_api_url: api.includes('VITE_API_BASE_URL') && read('.env.example').includes('/api/v1'),
  multipart_pdf_upload: api.includes('FormData') && api.includes("body.append('file', file)"),
  no_manual_content_type: !api.includes("'Content-Type': 'multipart/form-data'"),
  typed_api_contract: app.includes('ResumeAnalysisResponse') && detail.includes('JobMatch'),
  no_mock_runtime_results: !/mockMatches|fakeScore|Math\.random\(\)/.test(`${app}${api}`),
  upload_guards: app.includes('10 * 1024 * 1024') && upload.includes('application/pdf'),
  result_filtering: app.includes('Filter by role, company, or location'),
  model_disclaimer: detail.includes('not hiring probabilities'),
  error_state: app.includes('role="alert"') && api.includes('ApiError'),
  loading_state: app.includes('analysis-loading') && app.includes("state === 'loading'"),
  empty_filter_state: app.includes('No matching results'),
  responsive_breakpoints: ['1120px', '880px', '600px'].every((value) => css.includes(value)),
  reduced_motion: css.includes('prefers-reduced-motion'),
  keyboard_focus: css.includes(':focus-visible'),
  accessible_upload: upload.includes('aria-label="Choose PDF resume"'),
  external_link_safety: detail.includes('rel="noreferrer"'),
  score_bounded_for_display: read('src/components/ScoreRing.tsx').includes('Math.max(0, Math.min(100, score))'),
}

const passed = Object.values(checks).every(Boolean)
const report = {
  stage: 13,
  component: 'React frontend',
  status: passed ? 'passed' : 'failed',
  generated_at: new Date().toISOString(),
  checks,
  build_output_present: existsSync(resolve(root, 'dist/index.html')),
  notes: [
    'This static validator checks source-level integration and safety requirements.',
    'Run npm run build for TypeScript and production-bundle verification.',
    'Run the FastAPI backend before testing a real resume in the browser.',
  ],
}

const output = resolve(root, 'frontend_validation.json')
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`)
console.log(JSON.stringify(report, null, 2))
if (!passed) process.exitCode = 1
