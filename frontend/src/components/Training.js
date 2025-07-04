import React, { useState, useEffect } from 'react';
import Select from 'react-select'; // ADD THIS IMPORT
import { api } from '../services/api';

export function TrainTab({ contexts, uploadedData, onJobCreated }) {
  console.log('TrainTab rendered with uploadedData:', uploadedData);
  console.log('uploadedData contents:', JSON.stringify(uploadedData));
  const [config, setConfig] = useState({
    job_name: '',
    context: '', // Keep this for backward compatibility
    selected_contexts: [], // ADD THIS for multi-select
    task_type: 'binary_classification',
    epochs: 50,
    learning_rate: 0.001,
    batch_size: 32
  });
  const [training, setTraining] = useState(false);
  const [suggestions, setSuggestions] = useState(null);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [useCustomParams, setUseCustomParams] = useState(false);
  const [selectAll, setSelectAll] = useState(false); // ADD THIS

  // Prepare context options for react-select
  const contextOptions = contexts.map(ctx => ({ value: ctx, label: ctx })); // ADD THIS

  
  // Fetch parameter suggestions when data is uploaded
  useEffect(() => {
    console.log('useEffect triggered');
    console.log('uploadedData:', uploadedData);
    console.log('uploadedData?.data_id:', uploadedData?.data_id);
    
    if (uploadedData?.data_id) {
      console.log('Calling fetchSuggestions');
      fetchSuggestions();
    } else {
      console.log('No data_id, skipping fetchSuggestions');
    }
     // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadedData]);

  const fetchSuggestions = async () => {
    console.log('fetchSuggestions called');
    if (!uploadedData?.data_id) {
      console.log('No data_id in fetchSuggestions, returning');
      return;
    }
    
    console.log('Starting API call to analyze dataset with data_id:', uploadedData.data_id);
    setLoadingSuggestions(true);
    try {
      const result = await api.analyzeDataset(uploadedData.data_id);
      console.log('API response:', result);
      console.log('API response details:', JSON.stringify(result));
      setSuggestions(result);
      
      // Auto-fill suggested parameters if not using custom
      if (!useCustomParams && result.suggested_parameters) {
        setConfig(prev => ({
          ...prev,
          epochs: result.suggested_parameters.epochs || prev.epochs,
          learning_rate: result.suggested_parameters.learning_rate || prev.learning_rate,
          batch_size: result.suggested_parameters.batch_size || prev.batch_size
        }));
      }
    } catch (error) {
      console.error('fetchSuggestions error:', error);
      console.error('Error details:', error.message);
    } finally {
      setLoadingSuggestions(false);
    }
  };

  const applySuggestions = () => {
    if (suggestions?.suggested_parameters) {
      setConfig(prev => ({
        ...prev,
        epochs: suggestions.suggested_parameters.epochs || prev.epochs,
        learning_rate: suggestions.suggested_parameters.learning_rate || prev.learning_rate,
        batch_size: suggestions.suggested_parameters.batch_size || prev.batch_size
      }));
      setUseCustomParams(false);
    }
  };

  // ADD THIS: Handle multi-select context change
  const handleContextChange = (selected) => {
    setConfig(prev => ({
      ...prev,
      selected_contexts: selected || [],
      context: selected && selected.length > 0 ? selected[0].value : '' // Keep first for backward compatibility
    }));
    setSelectAll(false);
  };

  // ADD THIS: Handle select all contexts
  const handleSelectAll = () => {
    if (selectAll) {
      setConfig(prev => ({
        ...prev,
        selected_contexts: [],
        context: ''
      }));
      setSelectAll(false);
    } else {
      setConfig(prev => ({
        ...prev,
        selected_contexts: contextOptions,
        context: contextOptions[0]?.value || ''
      }));
      setSelectAll(true);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!uploadedData) {
      alert('Please upload training data first');
      return;
    }

    // MODIFIED: Check for selected contexts
    if (!config.selected_contexts || config.selected_contexts.length === 0) {
      alert('Please select at least one context');
      return;
    }

    setTraining(true);
    try {
      // MODIFIED: Include selected_contexts in the API call
      const trainingConfig = {
        ...config,
        selected_contexts: config.selected_contexts.map(ctx => ctx.value) // Extract values from react-select format
      };
      const result = await api.startTraining(uploadedData.data_id, trainingConfig);
      alert(`Training started! Job ID: ${result.job_id}`);
      onJobCreated();
      // Reset form
      setConfig({
        job_name: '',
        context: '',
        selected_contexts: [], // RESET THIS TOO
        task_type: 'binary_classification',
        epochs: 50,
        learning_rate: 0.001,
        batch_size: 32
      });
      setUseCustomParams(false);
      setSelectAll(false); // RESET THIS TOO
    } catch (error) {
      console.error('Training error:', error);
      alert('Training failed to start');
    } finally {
      setTraining(false);
    }
  };

  const handleParamChange = (field, value) => {
    setConfig({ ...config, [field]: value });
    setUseCustomParams(true);
  };

  // Log the current state for debugging
  console.log('Current state - suggestions:', suggestions, 'loadingSuggestions:', loadingSuggestions);

  return (
    <div className="tab-content">
      <h2>Configure Fine-Tuning</h2>
      
      {!uploadedData ? (
        <div className="warning">
          ⚠️ Please upload training data first
        </div>
      ) : (
        <>
          {/* Loading indicator */}
          {loadingSuggestions && (
            <div style={{ padding: '10px', background: '#e3f2fd', margin: '10px 0' }}>
              Analyzing dataset and generating parameter suggestions...
            </div>
          )}

          {/* Dataset Summary - Simplified */}
          {suggestions && (
            <div className="dataset-summary">
              <h3>Dataset Summary</h3>
              <div className="summary-stats">
                <span><strong>{suggestions.dataset_stats.total_samples}</strong> total proteins</span>
                <span className="separator">•</span>
                <span><strong>{suggestions.dataset_stats.positive_samples}</strong> positive</span>
                <span className="separator">•</span>
                <span><strong>{suggestions.dataset_stats.negative_samples}</strong> negative</span>
                <span className="separator">•</span>
                <span>Balance ratio: <strong>{suggestions.dataset_stats.imbalance_ratio}</strong></span>
              </div>
              
              {/* Warnings */}
              {suggestions.warnings && suggestions.warnings.length > 0 && (
                <div className="warnings-inline">
                  {suggestions.warnings.map((warning, idx) => (
                    <div key={idx} className={`warning-inline-item ${warning.level}`}>
                      <span className="warning-icon">
                        {warning.level === 'critical' ? '⚠️' : 'ℹ️'}
                      </span>
                      {warning.message}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Training Form */}
          <form onSubmit={handleSubmit} className="train-form">
            <div className="form-group">
              <label>Job Name:</label>
              <input
                type="text"
                value={config.job_name}
                onChange={(e) => setConfig({...config, job_name: e.target.value})}
                required
              />
            </div>

            {/* REPLACED: Single select with multi-select */}
            <div className="form-group">
              <label>Biological Contexts:</label>
              <button 
                type="button"
                onClick={handleSelectAll}
                style={{
                  marginBottom: '10px',
                  padding: '5px 15px',
                  backgroundColor: selectAll ? '#4CAF50' : '#f0f0f0',
                  color: selectAll ? 'white' : 'black',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                {selectAll ? '✓ All Contexts Selected' : 'Select All Contexts'}
              </button>
              
              <Select
                isMulti
                name="contexts"
                options={contextOptions}
                value={config.selected_contexts}
                onChange={handleContextChange}
                className="context-multi-select"
                classNamePrefix="select"
                placeholder="Select contexts..."
                required={config.selected_contexts.length === 0}
              />
              
              {config.selected_contexts.length > 0 && (
                <p style={{ marginTop: '5px', fontSize: '14px', color: '#666' }}>
                  {config.selected_contexts.length} context(s) selected
                </p>
              )}
            </div>

            <div className="form-group">
              <label>Task Type:</label>
              <select
                value={config.task_type}
                onChange={(e) => setConfig({...config, task_type: e.target.value})}
              >
                <option value="binary_classification">Binary Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>

            <div className="hyperparameters-section">
              <div className="hyperparameters-header">
                <h4>Hyperparameters</h4>
                {useCustomParams && (
                  <button 
                    type="button"
                    className="revert-btn"
                    onClick={applySuggestions}
                  >
                    ↻ Revert to Suggested Values
                  </button>
                )}
              </div>
              
              <div className="params-grid">
                <div className="param-group">
                  <label>
                    Epochs
                    {suggestions && !useCustomParams && (
                      <span className="suggested-tag">SUGGESTED</span>
                    )}
                  </label>
                  <input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                    min="1"
                    max="1000"
                  />
                  {suggestions && (
                    <p className="param-reason">{suggestions.suggested_parameters.epochs_reason}</p>
                  )}
                </div>

                <div className="param-group">
                  <label>
                    Learning Rate
                    {suggestions && !useCustomParams && (
                      <span className="suggested-tag">SUGGESTED</span>
                    )}
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={config.learning_rate}
                    onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                  />
                  {suggestions && (
                    <p className="param-reason">{suggestions.suggested_parameters.lr_reason}</p>
                  )}
                </div>

                <div className="param-group">
                  <label>
                    Batch Size
                    {suggestions && !useCustomParams && (
                      <span className="suggested-tag">SUGGESTED</span>
                    )}
                  </label>
                  <input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value))}
                    min="1"
                  />
                  {suggestions && (
                    <p className="param-reason">{suggestions.suggested_parameters.batch_size_reason}</p>
                  )}
                </div>
              </div>

              {/* Additional suggestions if available */}
              {suggestions?.suggested_parameters.model_type && (
                <div className="model-suggestion">
                  <p><strong>Suggested Model Architecture:</strong> {suggestions.suggested_parameters.model_type}</p>
                  <p className="param-reason">{suggestions.suggested_parameters.model_reason}</p>
                </div>
              )}
            </div>

            <button type="submit" disabled={training} className="train-button">
              {training ? 'Starting Training...' : 'Start Training'}
            </button>
          </form>
        </>
      )}
    </div>
  );
}