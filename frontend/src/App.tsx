import { useEffect, useMemo, useState } from 'react'
import './App.css'
import type {
  PerformanceEntry,
  PerformanceMode,
  RunResult,
  SuiteInfo,
  SuiteResponse,
  TestCase,
  TestScenario,
} from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'

type TestStatus = 'pending' | 'running' | 'passed' | 'blocked'

interface CustomRun {
  id: string
  testId: string
  mode: PerformanceMode
  threads: number
  durationMs: number
  notes?: string
  createdAt: string
}

interface ScenarioState {
  running: boolean
  result?: RunResult
}

interface ScenarioOverride {
  threads: string
  useCuda: boolean
  extraArgs: string
}

const statusOptions: { value: TestStatus; label: string }[] = [
  { value: 'pending', label: 'Pending' },
  { value: 'running', label: 'Running' },
  { value: 'passed', label: 'Passed' },
  { value: 'blocked', label: 'Blocked' },
]

const statusDescriptions: Record<TestStatus, string> = {
  pending: 'Awaiting execution',
  running: 'In progress',
  passed: 'Validated locally',
  blocked: 'Blocked (investigate before running)',
}

const FALLBACK_SUITE: SuiteInfo = {
  title: 'PRIR Parallel Analyzer Test Bench',
  description: 'Interactive dashboard for the OpenMP/MPI/CUDA log analyzer.',
  binaryPath: './build/bin/prir',
  logFile: 'access.log',
  notes: [],
}

const formatMs = (value?: number) => {
  if (!value && value !== 0) return '—'
  const seconds = value / 1000
  return `${seconds.toFixed(2)} s`
}

const buildPerfSummary = (entries?: PerformanceEntry[]) => {
  if (!entries || entries.length === 0) return null
  const cpuEntries = entries.filter((entry) => typeof entry.cpuMs === 'number')
  const gpuEntries = entries.filter((entry) => typeof entry.gpuMs === 'number')
  const cpuBest = cpuEntries.slice().sort((a, b) => (a.cpuMs ?? Infinity) - (b.cpuMs ?? Infinity))[0]
  const gpuBest = gpuEntries.slice().sort((a, b) => (a.gpuMs ?? Infinity) - (b.gpuMs ?? Infinity))[0]

  const baseCpu = cpuEntries.find((entry) => entry.threads === 1)?.cpuMs ?? cpuEntries[0]?.cpuMs
  const cpuSpeedup = baseCpu && cpuBest?.cpuMs ? baseCpu / cpuBest.cpuMs : undefined

  let bestGpuGain: { threads: number; deltaPct: number } | undefined
  entries.forEach((entry) => {
    if (!entry.cpuMs || !entry.gpuMs) return
    const deltaPct = ((entry.cpuMs - entry.gpuMs) / entry.cpuMs) * 100
    if (!bestGpuGain || deltaPct > bestGpuGain.deltaPct) {
      bestGpuGain = { threads: entry.threads, deltaPct }
    }
  })

  return { cpuBest, gpuBest, cpuSpeedup, bestGpuGain }
}

const scenarioKey = (testId: string, scenarioId: string) => `${testId}:${scenarioId}`

const commandHasFlag = (command: string, flag: string) => command.includes(flag)

const extractThreadsFromCommand = (command: string): string => {
  const match = command.match(/--threads(?:=|\s+)(\d+)/)
  return match ? match[1] : ''
}

const defaultScenarioOverride = (scenario: TestScenario): ScenarioOverride => ({
  threads: extractThreadsFromCommand(scenario.command),
  useCuda: scenario.perfTest
    ? false
    : scenario.mode === 'gpu' || commandHasFlag(scenario.command, '--use-cuda'),
  extraArgs: '',
})

function App() {
  const [suite, setSuite] = useState<SuiteInfo | null>(null)
  const [tests, setTests] = useState<TestCase[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [selectedTestId, setSelectedTestId] = useState('')
  const [statuses, setStatuses] = useState<Record<string, TestStatus>>({})
  const [scenarioStates, setScenarioStates] = useState<Record<string, ScenarioState>>({})
  const [scenarioOverrides, setScenarioOverrides] = useState<Record<string, ScenarioOverride>>({})
  const [scenarioErrors, setScenarioErrors] = useState<Record<string, string>>({})
  const [copiedScenario, setCopiedScenario] = useState('')
  const [modeInput, setModeInput] = useState<PerformanceMode>('cpu')
  const [threadsInput, setThreadsInput] = useState('8')
  const [durationInput, setDurationInput] = useState('')
  const [notesInput, setNotesInput] = useState('')
  const [formError, setFormError] = useState('')
  const [customRuns, setCustomRuns] = useState<Record<string, CustomRun[]>>({})

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setFetchError(null)
      try {
        const response = await fetch(`${API_BASE_URL}/tests`)
        if (!response.ok) {
          throw new Error(`Unable to load test suite (HTTP ${response.status}).`)
        }
        const payload: SuiteResponse = await response.json()
        setSuite(payload.suite)
        setTests(payload.tests)
        setStatuses((prev) => {
          const next: Record<string, TestStatus> = {}
          payload.tests.forEach((test) => {
            next[test.id] = prev[test.id] ?? 'pending'
          })
          return next
        })
        setSelectedTestId((current) => {
          if (current && payload.tests.some((test) => test.id === current)) {
            return current
          }
          return payload.tests[0]?.id ?? ''
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error.'
        setFetchError(message)
        setSuite(null)
        setTests([])
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const selectedTest = tests.length > 0 ? tests.find((test) => test.id === selectedTestId) ?? tests[0] : undefined
  const selectedTestRuns = selectedTest ? customRuns[selectedTest.id] ?? [] : []
  const perfSummary = useMemo(
      () => (selectedTest ? buildPerfSummary(selectedTest.performance) : null),
      [selectedTest],
  )
  const getScenarioOverride = (testId: string, scenario: TestScenario) => {
    const key = scenarioKey(testId, scenario.id)
    return scenarioOverrides[key] ?? defaultScenarioOverride(scenario)
  }

  const updateScenarioOverride = (testId: string, scenario: TestScenario, patch: Partial<ScenarioOverride>) => {
    const key = scenarioKey(testId, scenario.id)
    setScenarioOverrides((prev) => {
      const current = prev[key] ?? defaultScenarioOverride(scenario)
      return { ...prev, [key]: { ...current, ...patch } }
    })
    setScenarioErrors((prev) => {
      if (!prev[key]) return prev
      const next = { ...prev }
      delete next[key]
      return next
    })
  }

  const validateScenarioOverride = (override: ScenarioOverride) => {
    if (override.threads.trim().length === 0) return null
    const value = Number(override.threads)
    if (!Number.isFinite(value) || value <= 0) return 'Threads must be a positive number.'
    return null
  }

  const handleStatusChange = (testId: string, next: TestStatus) => {
    setStatuses((prev) => ({ ...prev, [testId]: next }))
  }

  const handleCopy = async (command: string, id: string) => {
    if (!navigator.clipboard) return
    try {
      await navigator.clipboard.writeText(command)
      setCopiedScenario(id)
      setTimeout(() => setCopiedScenario((current) => (current === id ? '' : current)), 1200)
    } catch {
      setCopiedScenario('')
    }
  }

  const handleRunScenario = async (test: TestCase, scenario: TestScenario) => {
    const key = scenarioKey(test.id, scenario.id)
    const override = getScenarioOverride(test.id, scenario)
    const validationError = validateScenarioOverride(override)
    if (validationError) {
      setScenarioErrors((prev) => ({ ...prev, [key]: validationError }))
      return
    }
    const payload: Record<string, unknown> = {}
    if (override.threads.trim()) payload.threads = Number(override.threads)
    if (!scenario.perfTest) {
      payload.useCuda = override.useCuda
    }
    if (override.extraArgs.trim()) payload.extraArgs = override.extraArgs.trim()
    setScenarioStates((prev) => ({ ...prev, [key]: { ...prev[key], running: true } }))
    try {
      const response = await fetch(
          `${API_BASE_URL}/tests/${encodeURIComponent(test.id)}/scenarios/${encodeURIComponent(scenario.id)}/run`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          },
      )
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        const detail = (body && body.detail) || `Run failed with HTTP ${response.status}`
        throw new Error(detail)
      }
      const result: RunResult = await response.json()
      setScenarioStates((prev) => ({ ...prev, [key]: { running: false, result } }))
    } catch (error) {
      const now = new Date().toISOString()
      const result: RunResult = {
        testId: test.id,
        scenarioId: scenario.id,
        command: scenario.command,
        exitCode: -1,
        durationMs: 0,
        stdout: '',
        stderr: '',
        startedAt: now,
        finishedAt: now,
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      }
      setScenarioStates((prev) => ({ ...prev, [key]: { running: false, result } }))
    }
  }

  const handleAddRun = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selectedTest) return
    const threads = Number(threadsInput)
    const durationSeconds = Number(durationInput)
    if (!Number.isFinite(threads) || threads <= 0) {
      setFormError('Threads must be a positive number.')
      return
    }
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      setFormError('Duration must be provided in seconds.')
      return
    }
    setFormError('')
    const entry: CustomRun = {
      id: `${selectedTest.id}-${Date.now()}`,
      testId: selectedTest.id,
      mode: modeInput,
      threads,
      durationMs: durationSeconds * 1000,
      notes: notesInput.trim() || undefined,
      createdAt: new Date().toISOString(),
    }
    setCustomRuns((prev) => {
      const existing = prev[selectedTest.id] ?? []
      return { ...prev, [selectedTest.id]: [entry, ...existing] }
    })
    setDurationInput('')
    setNotesInput('')
  }

  const handleClearRuns = () => {
    if (!selectedTest) return
    setCustomRuns((prev) => {
      if (!prev[selectedTest.id]) return prev
      const next = { ...prev }
      delete next[selectedTest.id]
      return next
    })
  }

  const activeSuite = suite ?? FALLBACK_SUITE

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <p className="eyebrow">PRIR regression harness</p>
          <h1>{activeSuite.title}</h1>
          <p className="suite-description">{activeSuite.description}</p>
        </div>
        <div className="suite-meta">
          <div>
            <span className="meta-label">Binary</span>
            <span className="meta-value">{activeSuite.binaryPath}</span>
          </div>
          <div>
            <span className="meta-label">Log source</span>
            <span className="meta-value">{activeSuite.logFile}</span>
          </div>
        </div>
        {activeSuite.notes.length > 0 && (
          <ul className="suite-notes">
            {activeSuite.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        )}
        {fetchError && <p className="error-banner">{fetchError}</p>}
      </header>

      <main className="app__body">
        {loading && (
          <div className="panel loading-panel">
            <p>Loading test definitions...</p>
          </div>
        )}

        {!loading && tests.length === 0 && (
          <div className="panel loading-panel">
            <p>No tests available. Start the backend (`uvicorn backend.main:app --reload`) and reload this page.</p>
          </div>
        )}

        {tests.length > 0 && selectedTest && (
          <>
            <aside className="test-list">
              {tests.map((test) => {
                const status = statuses[test.id]
                return (
                  <button
                    key={test.id}
                    className={`test-card ${selectedTest.id === test.id ? 'is-active' : ''}`}
                    onClick={() => setSelectedTestId(test.id)}
                  >
                    <div className="test-card__header">
                      <p className="test-card__eyebrow">{test.focus}</p>
                      <span className={`status-badge status-${status}`}>{statusDescriptions[status]}</span>
                    </div>
                    <h3>{test.name}</h3>
                    <p>{test.shortDescription}</p>
                    <div className="tag-list">
                      {test.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  </button>
                )
              })}
            </aside>

            <section className="test-details">
              <div className="panel test-summary">
                <div>
                  <h2>{selectedTest.name}</h2>
                  <p className="test-details__description">{selectedTest.details}</p>
                </div>
                <div className="test-summary__grid">
                  <div>
                    <span className="meta-label">Dataset</span>
                    <p className="meta-value">{selectedTest.dataset}</p>
                  </div>
                  <div>
                    <span className="meta-label">Phrases</span>
                    <p className="meta-value">{selectedTest.phrases.join(', ')}</p>
                  </div>
                  <div>
                    <span className="meta-label">Status</span>
                    <select
                      value={statuses[selectedTest.id] ?? 'pending'}
                      onChange={(event) => handleStatusChange(selectedTest.id, event.target.value as TestStatus)}
                    >
                      {statusOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <div className="panel scenarios">
                <h3>Test scenarios</h3>
                <div className="scenario-grid">
                  {selectedTest.scenarios.map((scenario) => {
                    const key = scenarioKey(selectedTest.id, scenario.id)
                    const state = scenarioStates[key]
                    const isRunning = state?.running ?? false
                    const lastResult = state?.result
                    const override = getScenarioOverride(selectedTest.id, scenario)
                    const overrideError = scenarioErrors[key]
                    return (
                      <article key={scenario.id} className="scenario-card">
                        <header>
                          <div className="scenario-label-row">
                            <p className="scenario-label">{scenario.label}</p>
                            {scenario.perfTest && <span className="scenario-badge">CPU & GPU</span>}
                          </div>
                          <div className="scenario-actions">
                            <button onClick={() => handleCopy(scenario.command, key)} className="copy-button" type="button">
                              {copiedScenario === key ? 'Copied' : 'Copy command'}
                            </button>
                            <button
                              type="button"
                              className="run-button"
                              onClick={() => handleRunScenario(selectedTest, scenario)}
                              disabled={isRunning}
                            >
                              {isRunning ? 'Running…' : 'Run scenario'}
                            </button>
                          </div>
                        </header>
                        <pre>{scenario.command}</pre>
                        <div className="expected-table">
                          {scenario.expectedValues.map((value) => (
                            <div key={value.label}>
                              <span>{value.label}</span>
                              <strong>{value.value}</strong>
                              {value.note && <p>{value.note}</p>}
                            </div>
                          ))}
                        </div>
                        {scenario.notes && <p className="scenario-note">{scenario.notes}</p>}
                        <div className="scenario-overrides">
                          <label>
                            Threads override
                            <input
                              type="number"
                              min={1}
                              step={1}
                              value={override.threads}
                              onChange={(event) =>
                                updateScenarioOverride(selectedTest.id, scenario, { threads: event.target.value })
                              }
                              placeholder="inherit"
                            />
                          </label>
                          {!scenario.perfTest ? (
                            <label className="checkbox-row">
                              <input
                                type="checkbox"
                                checked={override.useCuda}
                                onChange={(event) =>
                                  updateScenarioOverride(selectedTest.id, scenario, { useCuda: event.target.checked })
                                }
                              />
                              <span>Use CUDA</span>
                            </label>
                          ) : (
                            <div className="perf-test-hint">
                              CUDA and CPU runs execute sequentially in this mode.
                            </div>
                          )}
                          <label>
                            Extra CLI args
                            <input
                              type="text"
                              value={override.extraArgs}
                              onChange={(event) =>
                                updateScenarioOverride(selectedTest.id, scenario, { extraArgs: event.target.value })
                              }
                              placeholder="e.g. --from 2019-01-22T03:56:00"
                            />
                          </label>
                        </div>
                        {overrideError && <p className="scenario-error">{overrideError}</p>}
                        {lastResult && (
                          <div className={`run-result ${lastResult.success ? 'is-success' : 'is-failure'}`}>
                            <p>
                              Last run: exit {lastResult.exitCode} • {formatMs(lastResult.durationMs)} •{' '}
                              {new Date(lastResult.finishedAt).toLocaleString()}
                            </p>
                            <p className="run-command">
                              Command: <code>{lastResult.command}</code>
                            </p>
                            {lastResult.errorMessage && <p className="run-error">{lastResult.errorMessage}</p>}
                            <details>
                              <summary>stdout</summary>
                              <pre className="output-block">{lastResult.stdout || '(empty)'}</pre>
                            </details>
                            <details>
                              <summary>stderr</summary>
                              <pre className="output-block">{lastResult.stderr || '(empty)'}</pre>
                            </details>
                            {lastResult.perfTestSummary && (
                              <div className="perf-test-summary-block">
                                <p className="perf-test-summary__title">CPU vs GPU comparison</p>
                                {lastResult.perfTestSummary.entries.map((entry) => (
                                  <div key={entry.label} className="perf-test-summary__entry">
                                    <div className="perf-test-summary__entry-head">
                                      <strong>{entry.label}</strong>
                                      <span>{formatMs(entry.durationMs)}</span>
                                    </div>
                                    {entry.details && <p className="perf-test-summary__details">{entry.details}</p>}
                                    {entry.phrases.length > 0 && (
                                      <table className="perf-test-summary__table">
                                        <thead>
                                          <tr>
                                            <th>Phrase</th>
                                            <th>Count</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {entry.phrases.map((phrase) => (
                                            <tr key={phrase.phrase}>
                                              <td>{phrase.phrase}</td>
                                              <td>{phrase.count.toLocaleString()}</td>
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    )}
                                  </div>
                                ))}
                                {lastResult.perfTestSummary.warnings && lastResult.perfTestSummary.warnings.length > 0 && (
                                  <ul className="perf-test-summary__warnings">
                                    {lastResult.perfTestSummary.warnings.map((warning) => (
                                      <li key={warning}>{warning}</li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </article>
                    )
                  })}
                </div>
              </div>

              <div className="panel verification">
                <h3>Verification steps</h3>
                <ul>
                  {selectedTest.verification.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>

              {selectedTest.performance && (
                <div className="panel performance">
                  <div className="panel-header">
                    <h3>Time measurements</h3>
                    <p>Reference runs compare OpenMP (cpu) and CUDA histogram (gpu). Lower is better.</p>
                  </div>
                  <table>
                    <thead>
                      <tr>
                        <th>Threads</th>
                        <th>CPU (OpenMP)</th>
                        <th>GPU (CUDA)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedTest.performance.map((entry) => (
                        <tr key={entry.threads}>
                          <td>{entry.threads}</td>
                          <td>{formatMs(entry.cpuMs)}</td>
                          <td>{formatMs(entry.gpuMs)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {perfSummary && (
                    <div className="performance-summary">
                      {perfSummary.cpuBest && (
                        <p>
                          Fastest CPU run: <strong>{formatMs(perfSummary.cpuBest.cpuMs)}</strong> with{' '}
                          {perfSummary.cpuBest.threads} threads
                          {perfSummary.cpuSpeedup
                            ? ` (${perfSummary.cpuSpeedup.toFixed(2)}x vs single thread).`
                            : '.'}
                        </p>
                      )}
                      {perfSummary.gpuBest && (
                        <p>
                          Fastest GPU run: <strong>{formatMs(perfSummary.gpuBest.gpuMs)}</strong> with{' '}
                          {perfSummary.gpuBest.threads} threads.
                        </p>
                      )}
                      {perfSummary.bestGpuGain && (
                        <p>
                          Best GPU delta at {perfSummary.bestGpuGain.threads} threads:{' '}
                          {perfSummary.bestGpuGain.deltaPct.toFixed(2)}% vs CPU.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="panel manual-runs">
                <div className="panel-header">
                  <h3>Record your measurements</h3>
                  <p>Log the wall time from /usr/bin/time or perf and keep them next to the reference data.</p>
                </div>
                <form className="manual-form" onSubmit={handleAddRun}>
                  <label>
                    Mode
                    <select value={modeInput} onChange={(event) => setModeInput(event.target.value as PerformanceMode)}>
                      <option value="cpu">CPU (OpenMP)</option>
                      <option value="gpu">GPU (CUDA)</option>
                    </select>
                  </label>
                  <label>
                    Threads
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={threadsInput}
                      onChange={(event) => setThreadsInput(event.target.value)}
                    />
                  </label>
                  <label>
                    Duration (seconds)
                    <input
                      type="number"
                      min={0}
                      step="0.01"
                      value={durationInput}
                      onChange={(event) => setDurationInput(event.target.value)}
                      placeholder="9.47"
                    />
                  </label>
                  <label>
                    Notes (optional)
                    <input
                      value={notesInput}
                      onChange={(event) => setNotesInput(event.target.value)}
                      placeholder="Measured on laptop GPU"
                    />
                  </label>
                  {formError && <p className="form-error">{formError}</p>}
                  <button type="submit" disabled={!selectedTest}>
                    Add measurement
                  </button>
                </form>
                {selectedTestRuns.length > 0 && (
                  <div className="custom-runs">
                    <div className="custom-runs__header">
                      <h4>Recent runs</h4>
                      <button type="button" onClick={handleClearRuns} className="link-button">
                        Clear
                      </button>
                    </div>
                    <table>
                      <thead>
                        <tr>
                          <th>Mode</th>
                          <th>Threads</th>
                          <th>Duration</th>
                          <th>Notes</th>
                          <th>Timestamp</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedTestRuns.map((run) => (
                          <tr key={run.id}>
                            <td className={`mode-badge mode-${run.mode}`}>{run.mode.toUpperCase()}</td>
                            <td>{run.threads}</td>
                            <td>{formatMs(run.durationMs)}</td>
                            <td>{run.notes ?? '—'}</td>
                            <td>{new Date(run.createdAt).toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  )
}

export default App
