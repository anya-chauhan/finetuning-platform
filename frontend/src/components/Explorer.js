import React, { useState } from 'react';
import { api } from '../services/api';

export function ExploreTab({ contexts }) {
  const [activeExplorer, setActiveExplorer] = useState('similarity');
  
  return (
    <div className="tab-content">
      <h2>Explore Embeddings</h2>
      
      <div className="explorer-nav">
        <button 
          className={activeExplorer === 'similarity' ? 'active' : ''}
          onClick={() => setActiveExplorer('similarity')}
        >
          Protein Similarity
        </button>
        <button 
          className={activeExplorer === 'cross-context' ? 'active' : ''}
          onClick={() => setActiveExplorer('cross-context')}
        >
          Cross-Context Analysis
        </button>
        <button 
          className={activeExplorer === 'stats' ? 'active' : ''}
          onClick={() => setActiveExplorer('stats')}
        >
          Embedding Statistics
        </button>
      </div>
      
      {activeExplorer === 'similarity' && <SimilarityExplorer contexts={contexts} />}
      {activeExplorer === 'cross-context' && <CrossContextAnalysis contexts={contexts} />}
      {activeExplorer === 'stats' && <EmbeddingStats contexts={contexts} />}
    </div>
  );
}

function SimilarityExplorer({ contexts }) {
  const [queryProtein, setQueryProtein] = useState('');
  const [selectedContext, setSelectedContext] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleSearch = async () => {
    if (!queryProtein || !selectedContext) {
      alert('Please enter a protein ID and select a context');
      return;
    }

    setLoading(true);
    try {
      const data = await api.findSimilarProteins(queryProtein, selectedContext, topK);
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to find similar proteins');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="similarity-explorer">
      <div className="search-form">
        <input 
          type="text"
          placeholder="Enter protein ID"
          value={queryProtein}
          onChange={(e) => setQueryProtein(e.target.value)}
        />
        <select 
          value={selectedContext}
          onChange={(e) => setSelectedContext(e.target.value)}
        >
          <option value="">Select context...</option>
          {contexts.map(ctx => (
            <option key={ctx} value={ctx}>{ctx}</option>
          ))}
        </select>
        <input 
          type="number"
          value={topK}
          onChange={(e) => setTopK(e.target.value)}
          min="1"
          max="50"
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Find Similar Proteins'}
        </button>
      </div>
      
      {results && (
        <div className="results">
          <h3>Similar proteins to {results.query_protein}</h3>
          <div className="similarity-list">
            {results.similar_proteins.map((protein, i) => (
              <div key={i} className="similarity-item">
                <span>{protein.protein_id}</span>
                <div className="similarity-bar">
                  <div 
                    className="similarity-fill"
                    style={{ width: `${protein.similarity * 100}%` }}
                  />
                </div>
                <span>{(protein.similarity * 100).toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CrossContextAnalysis({ contexts }) {
  // Placeholder for cross-context analysis implementation
  return (
    <div className="cross-context-analysis">
      <p>Cross-context analysis coming soon...</p>
    </div>
  );
}

function EmbeddingStats({ contexts }) {
  // Placeholder for embedding statistics implementation
  return (
    <div className="embedding-stats">
      <p>Embedding statistics coming soon...</p>
    </div>
  );
}