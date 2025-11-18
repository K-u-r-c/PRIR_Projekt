export type PerformanceMode = 'cpu' | 'gpu'

export interface PerformanceEntry {
  threads: number
  cpuMs?: number
  gpuMs?: number
}

export interface ExpectedValue {
  label: string
  value: string
  note?: string
}

export interface TestScenario {
  id: string
  label: string
  mode?: PerformanceMode
  command: string
  expectedValues: ExpectedValue[]
  notes?: string
}

export interface TestCase {
  id: string
  name: string
  shortDescription: string
  details: string
  dataset: string
  focus: string
  phrases: string[]
  scenarios: TestScenario[]
  verification: string[]
  tags: string[]
  performance?: PerformanceEntry[]
}

export interface SuiteInfo {
  title: string
  description: string
  binaryPath: string
  logFile: string
  notes: string[]
}

export interface SuiteResponse {
  suite: SuiteInfo
  tests: TestCase[]
}

export interface RunResult {
  testId: string
  scenarioId: string
  command: string
  exitCode: number
  durationMs: number
  stdout: string
  stderr: string
  startedAt: string
  finishedAt: string
  success: boolean
  errorMessage?: string
}
