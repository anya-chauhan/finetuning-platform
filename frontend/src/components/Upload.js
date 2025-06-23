import React, { useState } from 'react';
import { api } from '../services/api';
import { getDataInsights } from '../utils/helpers';

export function UploadTab({ uploadedData, setUploadedData }) {
  const [uploading, setUploading] = useState(false);
  const [positiveFile, setPositiveFile] = useState(null);
  const [negativeFile, setNegativeFile] = useState(null);

  const handleFileUpload = async () => {
    if (!positiveFile || !negativeFile) {
      alert('Please select both positive and negative protein files');
      return;
    }
    
    setUploading(true);
    try {
      const data = await api.uploadTrainingData(positiveFile, negativeFile);
      setUploadedData(data);
    } catch (error) {
      console.error('Upload error:', error);
      alert(error.message);
    } finally {
      setUploading(false);
    }
  };

  const insights = getDataInsights(uploadedData);

  return (
    <div className="tab-content">
      <h2>Upload Fine-Tuning Dataset</h2>
      <p>Upload positive and negative protein JSON files (PINNACLE format)</p>
      
      <div className="upload-container">
        <div className="upload-section">
          <div className="file-input-group">
            <label htmlFor="positive-file">Positive Proteins (JSON):</label>
            <input
              id="positive-file"
              type="file"
              accept=".json"
              onChange={(e) => setPositiveFile(e.target.files[0])}
            />
            {positiveFile && <span className="file-name">✓ {positiveFile.name}</span>}
          </div>

          <div className="file-input-group">
            <label htmlFor="negative-file">Negative Proteins (JSON):</label>
            <input
              id="negative-file"
              type="file"
              accept=".json"
              onChange={(e) => setNegativeFile(e.target.files[0])}
            />
            {negativeFile && <span className="file-name">✓ {negativeFile.name}</span>}
          </div>

          <button 
            onClick={handleFileUpload}
            disabled={uploading || !positiveFile || !negativeFile}
            className="upload-button"
          >
            {uploading ? 'Uploading...' : 'Upload Files'}
          </button>

          {uploading && (
            <div className="uploading">
              <div className="spinner"></div>
              <p>Processing files...</p>
            </div>
          )}
        </div>

        {uploadedData && (
          <div className="upload-info-panel">
            <div className="upload-success">
              <h3>Data Loaded Successfully</h3>
              
              <div className="data-stats">
                <div className="stat-item">
                  <span className="stat-label">Total proteins:</span>
                  <span className="stat-value">{uploadedData.total_proteins}</span>
                </div>
                <div className="stat-item positive">
                  <span className="stat-label">Positive samples:</span>
                  <span className="stat-value">{uploadedData.positive_proteins}</span>
                </div>
                <div className="stat-item negative">
                  <span className="stat-label">Negative samples:</span>
                  <span className="stat-value">{uploadedData.negative_proteins}</span>
                </div>
              </div>

              <div className="data-insights">
                <h4>Dataset Insights</h4>
                <div className="insight-item">
                  <span>Positive/Negative ratio:</span>
                  <strong>{insights.ratio}</strong>
                </div>
                <div className="insight-item">
                  <span>{insights.recommendation}</span>
                </div>
                {uploadedData.duplicate_proteins > 0 && (
                  <div className="insight-item warning">
                    ⚠️ Found {uploadedData.duplicate_proteins} proteins in both sets
                  </div>
                )}
              </div>

              <div className="upload-timestamp">
                Uploaded: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}