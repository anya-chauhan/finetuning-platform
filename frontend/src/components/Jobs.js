import React, { useState, useEffect } from 'react';
import { getStatusColor } from '../utils/helpers';
import { api } from '../services/api'; // ADD THIS IMPORT

export function JobsTab({ jobs, onRefresh }) {
  const [selectedJob, setSelectedJob] = useState(null);

  useEffect(() => {
    const interval = setInterval(onRefresh, 2000); // Refresh every 2 seconds
    return () => clearInterval(interval);
  }, [onRefresh]);

  return (
    <div className="tab-content">
      <div className="jobs-header">
        <h2>Training Jobs</h2>
        <button onClick={onRefresh} className="refresh-button">
          🔄 Refresh
        </button>
      </div>

      <div className="jobs-list">
        {jobs.length === 0 ? (
          <p>No training jobs yet</p>
        ) : (
          jobs.map((job, index) => (
            <JobCard 
              key={index} 
              job={job} 
              onClick={() => setSelectedJob(job)}
            />
          ))
        )}
      </div>

      {selectedJob && (
        <JobDetails 
          job={selectedJob} 
          onClose={() => setSelectedJob(null)}
        />
      )}
    </div>
  );
}

function JobCard({ job, onClick }) {
  return (
    <div className="job-card" onClick={onClick}>
      <div className="job-header">
        <h3>{job.job_name}</h3>
        <span 
          className="status-badge"
          style={{ backgroundColor: getStatusColor(job.status) }}
        >
          {job.status}
        </span>
      </div>
      <p>Context{job.contexts && job.contexts.length > 1 ? 's' : ''}: {
      job.contexts 
        ? job.contexts.join(', ')  // This adds comma and space between contexts
        : job.context  // Fallback for old jobs
    }</p>
      <p>Progress: {job.progress}%</p>
      {job.status === 'started' && (
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${job.progress}%` }}
          ></div>
        </div>
      )}
    </div>
  );
}

function JobDetails({ job, onClose }) {
  const [showGeneImportance, setShowGeneImportance] = useState(false);
  const [geneImportance, setGeneImportance] = useState(null);
  const [loadingImportance, setLoadingImportance] = useState(false);

  const fetchGeneImportance = async () => {
    setLoadingImportance(true);
    try {
      // The gene importance should already be in the job object
      if (job.gene_importance) {
        setGeneImportance(job.gene_importance);
        setShowGeneImportance(true);
      } else {
        alert('Gene importance data not available for this job yet');
      }
    } catch (error) {
      console.error('Failed to load gene importance:', error);
      alert('Failed to load gene importance data');
    } finally {
      setLoadingImportance(false);
    }
  };

  // Helper function to format metric values
  const formatMetric = (value, isPercentage = false) => {
    if (value === null || value === undefined) return 'N/A';
    if (isPercentage) return `${(value * 100).toFixed(1)}%`;
    return value.toFixed(4);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Job Details: {job.job_name}</h3>
          <button onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <p><strong>Status:</strong> {job.status}</p>
          <p><strong>Context{job.contexts && job.contexts.length > 1 ? 's' : ''}:</strong> {
  job.contexts 
    ? job.contexts.join(', ')
    : job.context
}</p>
          <p><strong>Progress:</strong> {job.progress}%</p>
          <p><strong>Started:</strong> {new Date(job.timestamp).toLocaleString()}</p>
          
          {job.config && (
            <details>
              <summary>Configuration</summary>
              <pre>{JSON.stringify(job.config, null, 2)}</pre>
            </details>
          )}

          {job.metrics && job.metrics.length > 0 && (
            <details open>
              <summary>Training Metrics</summary>
              <div className="metrics">
                {/* Show latest metrics */}
                {(() => {
                  const latestMetric = job.metrics[job.metrics.length - 1];
                  return (
                    <div style={{ marginBottom: '15px' }}>
                      <h4 style={{ margin: '10px 0 5px 0' }}>Latest Metrics (Epoch {latestMetric.epoch}):</h4>
                      <div style={{ 
                        display: 'grid', 
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
                        gap: '10px',
                        backgroundColor: '#f5f5f5',
                        padding: '10px',
                        borderRadius: '5px'
                      }}>
                        <div>
                          <strong>Loss:</strong> {formatMetric(latestMetric.loss)}
                        </div>
                        {latestMetric.accuracy !== undefined && (
                          <div>
                            <strong>Accuracy:</strong> {formatMetric(latestMetric.accuracy, true)}
                          </div>
                        )}
                        {latestMetric.precision !== undefined && (
                          <div>
                            <strong>Precision:</strong> {formatMetric(latestMetric.precision, true)}
                          </div>
                        )}
                        {latestMetric.recall !== undefined && (
                          <div>
                            <strong>Recall:</strong> {formatMetric(latestMetric.recall, true)}
                          </div>
                        )}
                        {latestMetric.f1_score !== undefined && (
                          <div>
                            <strong>F1 Score:</strong> {formatMetric(latestMetric.f1_score)}
                          </div>
                        )}
                        {latestMetric.auc_roc !== undefined && (
                          <div>
                            <strong>AUC-ROC:</strong> {formatMetric(latestMetric.auc_roc)}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* Show best metrics if completed */}
                {job.status === 'completed' && job.best_metrics && (
                  <div style={{ marginBottom: '15px' }}>
                    <h4 style={{ margin: '10px 0 5px 0' }}>   Best Performance {job.best_metrics.epoch && `(Epoch ${job.best_metrics.epoch})`}:
    </h4>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
                      gap: '10px',
                      backgroundColor: '#e8f5e9',
                      padding: '10px',
                      borderRadius: '5px',
                      border: '1px solid #4caf50'
                    }}>
                      <div>
                        <strong>Loss:</strong> {formatMetric(job.best_metrics.loss)}
                      </div>
                      {job.best_metrics.accuracy !== undefined && (
                        <div>
                          <strong>Accuracy:</strong> {formatMetric(job.best_metrics.accuracy, true)}
                        </div>
                      )}
                      {job.best_metrics.precision !== undefined && (
                        <div>
                          <strong>Precision:</strong> {formatMetric(job.best_metrics.precision, true)}
                        </div>
                      )}
                      {job.best_metrics.recall !== undefined && (
                        <div>
                          <strong>Recall:</strong> {formatMetric(job.best_metrics.recall, true)}
                        </div>
                      )}
                      {job.best_metrics.f1_score !== undefined && (
                        <div>
                          <strong>F1 Score:</strong> {formatMetric(job.best_metrics.f1_score)}
                        </div>
                      )}
                      {job.best_metrics.auc_roc !== undefined && (
                        <div>
                          <strong>AUC-ROC:</strong> {formatMetric(job.best_metrics.auc_roc)}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Show epoch history */}
                <details>
                  <summary>Training History (Last 10 epochs)</summary>
                  <div style={{ marginTop: '10px' }}>
                    {job.metrics.slice(-10).map((metric, i) => (
                      <div key={i} style={{ 
                        padding: '5px 10px', 
                        backgroundColor: i % 2 === 0 ? '#f9f9f9' : 'white',
                        fontSize: '14px'
                      }}>
                        <strong>Epoch {metric.epoch}:</strong> 
                        Loss={formatMetric(metric.loss)}
                        {metric.accuracy !== undefined && `, Acc=${formatMetric(metric.accuracy, true)}`}
                        {metric.f1_score !== undefined && `, F1=${formatMetric(metric.f1_score)}`}
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            </details>
          )}

          {job.error && (
            <div className="error">
              <strong>Error:</strong> {job.error}
            </div>
          )}

          {/* Gene importance section */}
          {job.status === 'completed' && job.model_id && (
            <div className="gene-importance-section">
              {!showGeneImportance ? (
                <button 
                  onClick={fetchGeneImportance} 
                  disabled={loadingImportance}
                  className="view-importance-btn"
                  style={{
                    marginTop: '15px',
                    padding: '8px 16px',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  {loadingImportance ? 'Loading...' : '📊 View Gene Importance Rankings'}
                </button>
              ) : (
                <GeneImportanceDisplay 
                  geneImportance={geneImportance} 
                  onClose={() => setShowGeneImportance(false)}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// NEW COMPONENT FOR GENE IMPORTANCE DISPLAY
function GeneImportanceDisplay({ geneImportance, onClose }) {
  const [expandedContexts, setExpandedContexts] = useState({});

  const toggleContext = (context) => {
    setExpandedContexts(prev => ({
      ...prev,
      [context]: !prev[context]
    }));
  };

  const downloadCSV = (context, data) => {
    const csv = 'Rank,Gene,Importance Score\n' + 
      data.ranked_proteins.map((item, idx) => 
        `${idx + 1},${item[0]},${item[1].toFixed(4)}`
      ).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gene_importance_${context}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="gene-importance-container" style={{ marginTop: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4>Gene Importance Rankings by Context</h4>
        <button 
          onClick={onClose}
          style={{
            padding: '4px 8px',
            backgroundColor: '#f0f0f0',
            border: '1px solid #ddd',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Hide
        </button>
      </div>
      
      {Object.entries(geneImportance).map(([context, data]) => (
        <div 
          key={context} 
          className="context-section"
          style={{
            marginTop: '10px',
            border: '1px solid #e0e0e0',
            borderRadius: '6px',
            overflow: 'hidden'
          }}
        >
          <div 
            className="context-header"
            onClick={() => toggleContext(context)}
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '10px',
              backgroundColor: '#f5f5f5',
              cursor: 'pointer',
              userSelect: 'none'
            }}
          >
            <span style={{ marginRight: '10px', fontSize: '12px' }}>
              {expandedContexts[context] ? '▼' : '▶'}
            </span>
            <span style={{ flex: 1, fontWeight: '600' }}>{context}</span>
            <span style={{ marginRight: '15px', color: '#666', fontSize: '14px' }}>
              ({data.ranked_proteins.length} genes)
            </span>
            <button 
              className="download-btn"
              onClick={(e) => {
                e.stopPropagation();
                downloadCSV(context, data);
              }}
              style={{
                padding: '4px 8px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              📥 Download
            </button>
          </div>
          
          {expandedContexts[context] && (
            <div className="gene-list" style={{ padding: '15px', maxHeight: '400px', overflowY: 'auto' }}>
              <div 
                className="gene-list-header"
                style={{
                  display: 'grid',
                  gridTemplateColumns: '60px 200px 1fr',
                  padding: '10px 0',
                  borderBottom: '2px solid #e0e0e0',
                  fontWeight: '600',
                  color: '#666'
                }}
              >
                <span>Rank</span>
                <span>Gene</span>
                <span>Importance Score</span>
              </div>
              {data.ranked_proteins.slice(0, 50).map((item, idx) => (
                <div 
                  key={item[0]} 
                  className="gene-item"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '60px 200px 1fr',
                    padding: '8px 0',
                    borderBottom: '1px solid #f0f0f0',
                    alignItems: 'center'
                  }}
                >
                  <span style={{ color: '#666', fontSize: '14px' }}>#{idx + 1}</span>
                  <span style={{ fontWeight: '500', color: '#333' }}>{item[0]}</span>
                  <div style={{ position: 'relative', background: '#e0e0e0', height: '20px', borderRadius: '10px', overflow: 'hidden' }}>
                    <div 
                      className="score-bar"
                      style={{ 
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        height: '100%',
                        width: `${item[1] * 100}%`,
                        background: 'linear-gradient(to right, #4CAF50, #8BC34A)',
                        transition: 'width 0.3s'
                      }}
                    />
                    <span style={{
                      position: 'absolute',
                      right: '10px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {item[1].toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
              {data.ranked_proteins.length > 50 && (
                <div style={{ padding: '10px', textAlign: 'center', color: '#666', fontStyle: 'italic' }}>
                  ... and {data.ranked_proteins.length - 50} more genes
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}