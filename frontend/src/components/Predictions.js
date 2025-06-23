import React, { useState } from 'react';
import { api } from '../services/api';

export function PredictTab({ contexts, jobs }) {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedContext, setSelectedContext] = useState('');
  const [proteinIds, setProteinIds] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!selectedModel || !selectedContext || !proteinIds.trim()) {
      alert('Please fill all fields');
      return;
    }

    setLoading(true);
    try {
      const protein_list = proteinIds.split('\n').map(id => id.trim()).filter(id => id);
      const data = await api.predict(selectedModel, selectedContext, protein_list);
      setPredictions(data.predictions);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-content">
      <h2>Make Predictions</h2>
      
      {jobs.length === 0 ? (
        <div className="warning">
          ⚠️ No trained models available. Complete a training job first.
        </div>
      ) : (
        <form onSubmit={handlePredict} className="predict-form">
          <div className="form-group">
            <label>Select Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              required
            >
              <option value="">Choose a trained model...</option>
              {jobs.map((job, i) => (
                <option key={i} value={job.model_id}>
                  {job.job_name} (Context: {job.context})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Context for Prediction:</label>
            <select
              value={selectedContext}
              onChange={(e) => setSelectedContext(e.target.value)}
              required
            >
              <option value="">Select context...</option>
              {contexts.map(ctx => (
                <option key={ctx} value={ctx}>{ctx}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Protein IDs (one per line):</label>
            <textarea
              value={proteinIds}
              onChange={(e) => setProteinIds(e.target.value)}
              placeholder="PROTEIN_0001&#10;PROTEIN_0002&#10;PROTEIN_0003"
              rows="5"
              required
            />
          </div>

          <button type="submit" disabled={loading} className="predict-button">
            {loading ? 'Predicting...' : 'Make Predictions'}
          </button>
        </form>
      )}

      {predictions && (
        <div className="predictions-results">
          <h3>Prediction Results</h3>
          <div className="predictions-table">
            <div className="table-header">
              <span>Protein ID</span>
              <span>Prediction</span>
              <span>Confidence</span>
            </div>
            {predictions.map((pred, i) => (
              <div key={i} className="table-row">
                <span>{pred.protein_id}</span>
                <span>
                  {pred.prediction !== null ? pred.prediction.toFixed(4) : 'Error'}
                </span>
                <span>
                  {pred.confidence ? pred.confidence.toFixed(4) : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}