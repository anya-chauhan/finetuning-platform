import React, { useState, useEffect } from 'react';
import { api } from './services/api';
import { UploadTab } from './components/Upload';
import { TrainTab } from './components/Training';
import { JobsTab } from './components/Jobs';
import { PredictTab } from './components/Predictions';
import { ExploreTab } from './components/Explorer';

import './styles/variables.css';
import './styles/index.css';
import './styles/components/Header.css';
import './styles/components/Navigation.css';
import './styles/components/Upload.css';
import './styles/components/Forms.css';
import './styles/components/Jobs.css';
import './styles/components/Modal.css';
import './styles/components/Explorer.css';
import './styles/components/Predictions.css';
import './styles/utilities/animations.css';
import './styles/utilities/responsive.css';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [contexts, setContexts] = useState([]);
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [uploadedData, setUploadedData] = useState(null);

  useEffect(() => {
    fetchContexts();
    fetchJobs();
  }, []);

  const fetchContexts = async () => {
    try {
      const data = await api.fetchContexts();
      setContexts(data);
    } catch (error) {
      console.error('Error fetching contexts:', error);
    }
  };

  const fetchJobs = async () => {
    try {
      const jobs = await api.fetchJobs();
      setTrainingJobs(jobs);
    } catch (error) {
      console.error('Error fetching jobs:', error);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Protein Fine-Tuning Platform</h1>
        <p>Web-based fine-tuning of pre-trained protein embeddings for biological insights</p>
      </header>

      <nav className="nav">
        <button 
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Upload Dataset
        </button>
        <button 
          className={activeTab === 'train' ? 'active' : ''}
          onClick={() => setActiveTab('train')}
        >
          Run Fine-tuning
        </button>
        <button 
          className={activeTab === 'jobs' ? 'active' : ''}
          onClick={() => setActiveTab('jobs')}
        >
          Jobs Dashboard
        </button>
        <button 
          className={activeTab === 'predict' ? 'active' : ''}
          onClick={() => setActiveTab('predict')}
        >
          Make Predictions
        </button>
        <button 
          className={activeTab === 'explore' ? 'active' : ''}
          onClick={() => setActiveTab('explore')}
        >
          Explore Embeddings
        </button>
      </nav>

      <main className="main">
        {activeTab === 'upload' && (
          <UploadTab 
            uploadedData={uploadedData} 
            setUploadedData={setUploadedData} 
          />
        )}
        {activeTab === 'train' && (
          <TrainTab 
            contexts={contexts} 
            uploadedData={uploadedData}
            onJobCreated={fetchJobs}
          />
        )}
        {activeTab === 'jobs' && (
          <JobsTab 
            jobs={trainingJobs} 
            onRefresh={fetchJobs}
          />
        )}
        {activeTab === 'predict' && (
          <PredictTab 
            contexts={contexts}
            jobs={trainingJobs.filter(job => job.status === 'completed')}
          />
        )}
        {activeTab === 'explore' && (
          <ExploreTab contexts={contexts} />
        )}
      </main>
      
      <footer className="footer">
        <div className="footer-content">
        </div>
      </footer>
    </div>
  );
}

export default App;