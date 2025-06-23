import React, { useState, useEffect } from 'react';
import { getStatusColor } from '../utils/helpers';

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
      <p>Context: {job.context}</p>
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
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Job Details: {job.job_name}</h3>
          <button onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <p><strong>Status:</strong> {job.status}</p>
          <p><strong>Context:</strong> {job.context}</p>
          <p><strong>Progress:</strong> {job.progress}%</p>
          <p><strong>Started:</strong> {new Date(job.timestamp).toLocaleString()}</p>
          
          {job.config && (
            <details>
              <summary>Configuration</summary>
              <pre>{JSON.stringify(job.config, null, 2)}</pre>
            </details>
          )}

          {job.metrics && job.metrics.length > 0 && (
            <details>
              <summary>Training Metrics</summary>
              <div className="metrics">
                {job.metrics.slice(-5).map((metric, i) => (
                  <p key={i}>Epoch {metric.epoch}: Loss = {metric.loss.toFixed(4)}</p>
                ))}
              </div>
            </details>
          )}

          {job.error && (
            <div className="error">
              <strong>Error:</strong> {job.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}