// Utility functions for the Protein Fine-Tuning Platform

export const getDataInsights = (uploadedData) => {
  if (!uploadedData) return null;
  
  const ratio = uploadedData.positive_proteins / uploadedData.negative_proteins;
  const isBalanced = ratio > 0.8 && ratio < 1.2;
  
  return {
    ratio: ratio.toFixed(2),
    isBalanced,
    recommendation: isBalanced 
      ? "✅ Well-balanced dataset" 
      : ratio > 1 
        ? "⚠️ More positive samples - consider balancing" 
        : "⚠️ More negative samples - consider balancing"
  };
};

export const getStatusColor = (status) => {
  const colors = {
    'completed': '#4CAF50',
    'failed': '#f44336',
    'started': '#2196F3'
  };
  return colors[status] || '#757575';
};

export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleString();
};

export const formatDuration = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
};

export const validateProteinId = (proteinId) => {
  // Add any specific validation rules for protein IDs
  return proteinId && proteinId.trim().length > 0;
};

export const parseProteinList = (text) => {
  return text
    .split('\n')
    .map(id => id.trim())
    .filter(id => validateProteinId(id));
};