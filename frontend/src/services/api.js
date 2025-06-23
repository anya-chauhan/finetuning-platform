const API_BASE = 'http://localhost:8000';

export const api = {
  async fetchContexts() {
    const response = await fetch(`${API_BASE}/contexts`);
    const data = await response.json();
    return data.contexts;
  },

  async fetchJobs() {
    const response = await fetch(`${API_BASE}/jobs`);
    const data = await response.json();
    return Object.values(data.jobs);
  },

  async uploadTrainingData(positiveFile, negativeFile) {
    const formData = new FormData();
    formData.append('positive_file', positiveFile);
    formData.append('negative_file', negativeFile);

    const response = await fetch(`${API_BASE}/upload-training-data`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Upload failed: ${error}`);
    }

    return response.json();
  },

  async startTraining(dataId, config) {
    const response = await fetch(`${API_BASE}/train?data_id=${dataId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    if (!response.ok) {
      throw new Error('Training failed to start');
    }

    return response.json();
  },

  async predict(modelId, context, proteinIds) {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: modelId,
        context: context,
        protein_ids: proteinIds
      })
    });

    if (!response.ok) {
      throw new Error('Prediction failed');
    }

    return response.json();
  },

  async findSimilarProteins(proteinId, context, topK) {
    const response = await fetch(
      `${API_BASE}/embeddings/similar-proteins?protein_id=${proteinId}&context=${context}&top_k=${topK}`
    );

    if (!response.ok) {
      throw new Error('Failed to find similar proteins');
    }

    return response.json();
  },

  analyzeDataset: async (dataId) => {
    const response = await fetch(`${API_BASE}/training/analyze/${dataId}`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to analyze dataset');
    return response.json();
  },
};
