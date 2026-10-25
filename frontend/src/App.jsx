import { useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  getCandidates,
  getDashboard,
  getEntity,
  getMatches,
  getModels,
  getSubmission,
  searchEntities,
} from './api'

const navItems = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'entity', label: 'Entity Match' },
  { key: 'models', label: 'Models' },
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'submission', label: 'Submission' },
]

const formatMetric = (value) => (typeof value === 'number' ? value.toFixed(4) : value)
const percent = (value) => `${Number(value || 0).toFixed(2)}%`

function StatCard({ label, value, tone = 'default' }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  )
}

function ModelBar({ value, max }) {
  return (
    <div className="bar-track">
      <div className="bar-fill" style={{ width: `${(value / max) * 100}%` }} />
    </div>
  )
}

function App() {
  const [activePage, setActivePage] = useState('dashboard')
  const [dashboard, setDashboard] = useState(null)
  const [models, setModels] = useState([])
  const [submission, setSubmission] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [selectedEntityId, setSelectedEntityId] = useState('')
  const [entity, setEntity] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [matches, setMatches] = useState([])
  const [matchSourceTab, setMatchSourceTab] = useState('Source 2')
  const [loading, setLoading] = useState({ dashboard: true, entity: false })
  const [featureOpen, setFeatureOpen] = useState({})

  useEffect(() => {
    Promise.all([getDashboard(), getModels(), getSubmission()])
      .then(([dashboardData, modelData, submissionData]) => {
        setDashboard(dashboardData)
        setModels(modelData)
        setSubmission(submissionData)
      })
      .finally(() => setLoading((current) => ({ ...current, dashboard: false })))
  }, [])

  useEffect(() => {
    const trimmed = searchQuery.trim()
    if (!trimmed) {
      setSearchResults([])
      return
    }

    const timer = setTimeout(() => {
      searchEntities(trimmed)
        .then((results) => setSearchResults(results))
        .catch(() => setSearchResults([]))
    }, 200)

    return () => clearTimeout(timer)
  }, [searchQuery])

  useEffect(() => {
    if (!selectedEntityId) return
    setLoading((current) => ({ ...current, entity: true }))
    Promise.all([getEntity(selectedEntityId), getCandidates(selectedEntityId), getMatches(selectedEntityId)])
      .then(([entityData, candidateData, matchData]) => {
        setEntity(entityData)
        setCandidates(candidateData)
        setMatches(matchData)
      })
      .catch(() => {
        setEntity(null)
        setCandidates([])
        setMatches([])
      })
      .finally(() => setLoading((current) => ({ ...current, entity: false })))
  }, [selectedEntityId])

  const filteredMatches = useMemo(() => {
    return (matches || []).filter((item) => item.source === matchSourceTab)
  }, [matches, matchSourceTab])

  const sourceSection = useMemo(() => {
    return {
      'Source 2': filteredMatches.filter((match) => match.source === 'Source 2'),
      'Source 3': filteredMatches.filter((match) => match.source === 'Source 3'),
    }
  }, [filteredMatches])

  const maxModelMetric = useMemo(() => {
    const values = (models || []).map((entry) => Number(entry.f05 || 0))
    return values.length ? Math.max(...values) : 1
  }, [models])

  const renderDashboard = () => (
    <div className="page-view">
      <header className="hero-header">
        <div>
          <h1>Business Entity Resolution</h1>
          <p>AI-powered matching of noisy business records across multiple data sources.</p>
        </div>
        <div className="status-pill success">System OK</div>
      </header>

      <section className="dataset-section">
        <h2>Training</h2>
        <div className="stats-grid">
          <StatCard label="Source 1 Entities" value={dashboard?.source1_train_count?.toLocaleString() ?? '—'} tone="primary" />
          <StatCard label="Source 2 Entities" value={dashboard?.source2_train_count?.toLocaleString() ?? '—'} tone="cyan" />
          <StatCard label="Source 3 Entities" value={dashboard?.source3_train_count?.toLocaleString() ?? '—'} tone="violet" />
          <StatCard label="Candidate Pairs" value={dashboard?.training_candidate_count?.toLocaleString() ?? '—'} tone="orange" />
          <StatCard label="Blocking Recall" value={dashboard ? percent(dashboard.blocking_recall * 100) : '—'} tone="green" />
          <StatCard label="Search Space Reduction" value={dashboard ? percent(dashboard.search_space_reduction) : '—'} tone="amber" />
        </div>
      </section>

      <section className="dataset-section">
        <h2>Official Test</h2>
        <div className="stats-grid">
          <StatCard label="Source 1 Entities" value={dashboard?.source1_test_count?.toLocaleString() ?? '—'} tone="primary" />
          <StatCard label="Source 2 Entities" value={dashboard?.source2_test_count?.toLocaleString() ?? '—'} tone="cyan" />
          <StatCard label="Source 3 Entities" value={dashboard?.source3_test_count?.toLocaleString() ?? '—'} tone="violet" />
          <StatCard label="Candidate Pairs" value={dashboard?.test_candidate_count?.toLocaleString() ?? '—'} tone="orange" />
          <StatCard label="Predicted Matches" value={dashboard?.test_predicted_matches?.toLocaleString() ?? '—'} tone="green" />
        </div>
        <p className="helper-note">Test candidate recall is not measured because TEST ground truth is unavailable.</p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Model Performance</h2>
        </div>
        <div className="panel-body">
          <div className="mini-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F0.5</th>
                  <th>PR-AUC</th>
                </tr>
              </thead>
              <tbody>
                {(models || []).map((model) => (
                  <tr key={model.name}>
                    <td>{model.name}</td>
                    <td>{formatMetric(model.precision)}</td>
                    <td>{formatMetric(model.recall)}</td>
                    <td>{formatMetric(model.f05)}</td>
                    <td>{formatMetric(model.pr_auc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="panel selected-model-panel">
        <div className="panel-header">
          <h2>Validation Metrics</h2>
        </div>
        <div className="two-col">
          <div className="selected-model-card">
            <span className="eyebrow">Final Model</span>
            <strong>{dashboard?.selected_model || 'Gradient Boosting'}</strong>
          </div>
          <div className="selected-model-card">
            <span className="eyebrow">Final Threshold</span>
            <strong>{dashboard?.selected_threshold ?? 0.94}</strong>
          </div>
        </div>
      </section>
    </div>
  )

  const renderEntityExplorer = () => (
    <div className="page-view">
      <header className="panel compact-panel">
        <div className="panel-header">
          <h2>Entity Match Explorer</h2>
        </div>
        <div className="search-wrap">
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search Source 1 business..."
          />
        </div>
        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((row) => (
              <button key={row.entity_id} type="button" onClick={() => setSelectedEntityId(row.entity_id)}>
                <span>{row.entity_id}</span>
                <strong>{row.business_name}</strong>
                <small>{row.country}</small>
              </button>
            ))}
          </div>
        )}
      </header>

      {loading.entity ? (
        <div className="loading-box">Loading entity data...</div>
      ) : entity ? (
        <>
          <section className="panel entity-panel">
            <div className="panel-header">
              <h3>Source 1</h3>
            </div>
            <div className="entity-card-grid">
              <div><span className="field-label">Entity ID</span><strong>{entity.entity_id}</strong></div>
              <div><span className="field-label">Business Name</span><strong>{entity.business_name}</strong></div>
              <div><span className="field-label">Address</span><strong>{entity.business_address}</strong></div>
              <div><span className="field-label">Country</span><strong>{entity.country}</strong></div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header fit-between">
              <h3>Candidate Matches</h3>
              <div className="toggle-tabs">
                {['Source 2', 'Source 3'].map((source) => (
                  <button
                    key={source}
                    type="button"
                    className={source === matchSourceTab ? 'active' : ''}
                    onClick={() => setMatchSourceTab(source)}
                  >
                    {source}
                  </button>
                ))}
              </div>
            </div>
            <div className="match-grid">
              {(sourceSection[matchSourceTab] || []).map((match) => (
                <article key={`${match.source}-${match.candidate_entity_id}`} className="match-card">
                  <div className="match-head">
                    <div>
                      <div className="match-id">{match.candidate_entity_id}</div>
                      <div className="candidate-source">{match.source}</div>
                      <h4>{match.business_name || 'Unknown Business'}</h4>
                    </div>
                    <span className={`decision-badge ${match.is_match ? 'match' : 'no-match'}`}>
                      {match.is_match ? '✓ MATCH' : '✕ NO MATCH'}
                    </span>
                  </div>
                  <div className="meta-row">
                    <span>{match.business_address}</span>
                  </div>
                  <div className="meta-row">
                    <span>{match.country}</span>
                    <strong>{(match.probability * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="meta-row">
                    <span>Match Probability</span>
                    <span>{(match.probability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="feature-toggle">
                    <button type="button" onClick={() => setFeatureOpen((current) => ({ ...current, [match.candidate_entity_id]: !current[match.candidate_entity_id] }))}>
                      {featureOpen[match.candidate_entity_id] ? 'Hide Features' : 'View Features'}
                    </button>
                  </div>
                  {featureOpen[match.candidate_entity_id] && (
                    <div className="feature-panel">
                      <div className="feature-row"><span>Name Similarity</span><strong>{match.features.name_jaccard ?? '—'}</strong></div>
                      <div className="feature-row"><span>Address Similarity</span><strong>{match.features.address_jaccard ?? '—'}</strong></div>
                      <div className="feature-row"><span>Country Match</span><strong>{match.features.country_exact ?? '—'}</strong></div>
                    </div>
                  )}
                </article>
              ))}
            </div>
          </section>
        </>
      ) : (
        <div className="empty-state">Select a Source 1 entity to inspect matching candidates.</div>
      )}
    </div>
  )

  const renderModels = () => (
    <div className="page-view">
      <header className="hero-header tight">
        <div>
          <h1>Model Comparison</h1>
          <p>Validation results from the held-out training split.</p>
        </div>
        <div className="status-pill solid">Selection criterion: validation F0.5</div>
      </header>

      <div className="two-col model-selection">
        <div className="selected-model-card">
          <span className="eyebrow">Selected Model</span>
          <strong>{dashboard?.selected_model || 'Gradient Boosting'}</strong>
        </div>
        <div className="selected-model-card">
          <span className="eyebrow">Selected Threshold</span>
          <strong>{dashboard?.selected_threshold ?? 0.94}</strong>
        </div>
      </div>

      <div className="model-metric-grid">
        {(models || []).map((model) => (
          <div key={model.name} className={`metric-card ${dashboard?.selected_model === model.name ? 'selected' : ''}`}>
            <div className="metric-header">
              <strong>{model.name}</strong>
              {dashboard?.selected_model === model.name && <span className="selected-dot">Selected</span>}
            </div>
            <div className="metric-list">
              <div><span>Precision</span><strong>{formatMetric(model.precision)}</strong></div>
              <div><span>Recall</span><strong>{formatMetric(model.recall)}</strong></div>
              <div><span>F0.5</span><strong>{formatMetric(model.f05)}</strong></div>
              <div><span>PR-AUC</span><strong>{formatMetric(model.pr_auc)}</strong></div>
              <div><span>FP</span><strong>{model.false_positives}</strong></div>
              <div><span>FN</span><strong>{model.false_negatives}</strong></div>
            </div>
          </div>
        ))}
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Comparison Table</h2>
        </div>
        <div className="panel-body">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F0.5</th>
                <th>PR-AUC</th>
                <th>FP</th>
                <th>FN</th>
              </tr>
            </thead>
            <tbody>
              {(models || []).map((model) => (
                <tr key={model.name} className={dashboard?.selected_model === model.name ? 'selected-row' : ''}>
                  <td>{model.name}</td>
                  <td>{formatMetric(model.precision)}</td>
                  <td>{formatMetric(model.recall)}</td>
                  <td>{formatMetric(model.f05)}</td>
                  <td>{formatMetric(model.pr_auc)}</td>
                  <td>{model.false_positives}</td>
                  <td>{model.false_negatives}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )

  const pipelineStages = [
    'Raw Data',
    'Cleaning',
    'Normalization',
    'Candidate Generation',
    'Feature Engineering',
    'Model Prediction',
    'Threshold',
    'Entity Matches',
    'Submission',
  ]

  const renderPipeline = () => (
    <div className="page-view">
      <header className="hero-header tight">
        <div>
          <h1>ML Pipeline</h1>
          <p>Same project pipeline used by the trained model and threshold.</p>
        </div>
      </header>
      <div className="pipeline-steps">
        {pipelineStages.map((step, index) => (
          <div key={step} className="pipeline-step">
            <div className="pipeline-node">{step}</div>
            {index < pipelineStages.length - 1 && <div className="pipeline-arrow">↓</div>}
          </div>
        ))}
      </div>
      <div className="panel">
        <div className="panel-header">
          <h2>Project Statistics</h2>
        </div>
        <div className="panel-body pipeline-copy">
          <div>
            <strong>Candidate Generation</strong>
            <p>80M possible comparisons → 21.6M candidate comparisons → 72.96% search-space reduction.</p>
          </div>
          <div>
            <strong>Blocking Metric</strong>
            <p>100% blocking recall on training data, using the same blocking strategy as the real project pipeline.</p>
          </div>
        </div>
      </div>
    </div>
  )

  const renderSubmission = () => (
    <div className="page-view">
      <header className="hero-header tight">
        <div>
          <h1>Submission</h1>
          <p>Generated from the final TEST predictions.</p>
        </div>
      </header>

      <div className="panel submission-panel">
        <div className="panel-header fit-between">
          <h2>Submission Status</h2>
          <span className="status-pill muted">
            {submission?.available ? 'Output Files Available' : 'Output Files Missing'}
          </span>
        </div>
        <div className="submission-grid">
          <div><span>Total Source 1 Test Entities</span><strong>{submission?.total_source1_entities ?? '—'}</strong></div>
          <div><span>Entities With Matches</span><strong>{submission?.entities_with_matches ?? '—'}</strong></div>
          <div><span>Entities Without Matches</span><strong>{submission?.entities_without_matches ?? '—'}</strong></div>
          <div><span>Total Matches</span><strong>{submission?.total_matches ?? '—'}</strong></div>
          <div><span>Source 2 Matches</span><strong>{submission?.source2_matches ?? '—'}</strong></div>
          <div><span>Source 3 Matches</span><strong>{submission?.source3_matches ?? '—'}</strong></div>
          <div><span>Total Candidate Pairs</span><strong>{submission?.total_candidate_pairs?.toLocaleString() ?? '—'}</strong></div>
        </div>
        <div className="status-row">
          <span>Official Validator</span>
          <strong>{submission?.official_validator_status === 'NOT_AVAILABLE' ? 'NOT AVAILABLE' : submission?.official_validator_status ?? 'NOT AVAILABLE'}</strong>
        </div>
        <p className="submission-note">The official challenge validator was not included in the provided project workspace, so official validation could not be executed.</p>
      </div>
    </div>
  )

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">BER</div>
          <div>
            <div className="brand-name">Business Entity Resolution</div>
            <div className="brand-subtitle">System OK</div>
          </div>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <button key={item.key} type="button" className={activePage === item.key ? 'nav-item active' : 'nav-item'} onClick={() => setActivePage(item.key)}>
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main-panel">
        {activePage === 'dashboard' && renderDashboard()}
        {activePage === 'entity' && renderEntityExplorer()}
        {activePage === 'models' && renderModels()}
        {activePage === 'pipeline' && renderPipeline()}
        {activePage === 'submission' && renderSubmission()}
      </main>
    </div>
  )
}

export default App
