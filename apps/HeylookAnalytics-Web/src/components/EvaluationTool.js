import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, Plus, Save, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import './EvaluationTool.css';

const API_BASE = 'http://localhost:8080';

function EvaluationTool() {
  const [evalSets, setEvalSets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedSet, setSelectedSet] = useState(null);
  const [createMode, setCreateMode] = useState(false);
  const [runningEval, setRunningEval] = useState(null);
  const [evalResults, setEvalResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New eval set form
  const [newEvalName, setNewEvalName] = useState('');
  const [newEvalDescription, setNewEvalDescription] = useState('');
  const [newEvalPrompts, setNewEvalPrompts] = useState([
    { prompt: '', expected_contains: [], tags: [] }
  ]);
  
  // Run configuration
  const [selectedModels, setSelectedModels] = useState([]);
  const [iterations, setIterations] = useState(1);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);

  useEffect(() => {
    fetchEvalSets();
    fetchModels();
  }, []);

  const fetchEvalSets = async () => {
    try {
      const response = await axios.get(`${API_BASE}/v1/eval/list`);
      setEvalSets(response.data.evaluation_sets || []);
    } catch (err) {
      console.error('Error fetching evaluation sets:', err);
      if (err.response?.status !== 503) {
        setError('Failed to fetch evaluation sets');
      }
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/v1/models`);
      setModels(response.data.data || []);
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  };

  const addPrompt = () => {
    setNewEvalPrompts([...newEvalPrompts, { prompt: '', expected_contains: [], tags: [] }]);
  };

  const removePrompt = (index) => {
    setNewEvalPrompts(newEvalPrompts.filter((_, i) => i !== index));
  };

  const updatePrompt = (index, field, value) => {
    const updated = [...newEvalPrompts];
    updated[index][field] = value;
    setNewEvalPrompts(updated);
  };

  const createEvalSet = async () => {
    if (!newEvalName || newEvalPrompts.length === 0) {
      setError('Name and at least one prompt are required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/v1/eval/create`, {
        name: newEvalName,
        description: newEvalDescription,
        prompts: newEvalPrompts.filter(p => p.prompt.trim())
      });

      await fetchEvalSets();
      setCreateMode(false);
      resetForm();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create evaluation set');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setNewEvalName('');
    setNewEvalDescription('');
    setNewEvalPrompts([{ prompt: '', expected_contains: [], tags: [] }]);
  };

  const runEvaluation = async () => {
    if (!selectedSet || selectedModels.length === 0) {
      setError('Please select an evaluation set and at least one model');
      return;
    }

    setLoading(true);
    setError(null);
    setEvalResults(null);

    try {
      const response = await axios.post(`${API_BASE}/v1/eval/run`, {
        eval_id: selectedSet.eval_id,
        models: selectedModels,
        iterations: iterations,
        parameters: {
          temperature: temperature,
          max_tokens: maxTokens
        }
      });

      setRunningEval(response.data);
      
      // Poll for results
      pollEvalStatus(response.data.run_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to run evaluation');
      setLoading(false);
    }
  };

  const pollEvalStatus = async (runId) => {
    const poll = async () => {
      try {
        const response = await axios.get(`${API_BASE}/v1/eval/run/${runId}`);
        
        if (response.data.status === 'completed') {
          setEvalResults(response.data);
          setRunningEval(null);
          setLoading(false);
        } else if (response.data.status === 'failed') {
          setError('Evaluation failed');
          setRunningEval(null);
          setLoading(false);
        } else {
          setRunningEval(response.data);
          setTimeout(poll, 1000); // Poll every second
        }
      } catch (err) {
        setError('Failed to fetch evaluation status');
        setLoading(false);
      }
    };
    
    poll();
  };

  const renderResults = () => {
    if (!evalResults || !evalResults.results) return null;

    const models = Object.keys(evalResults.results);
    
    return (
      <div className="eval-results">
        <h3>Evaluation Results</h3>
        
        <div className="results-summary">
          <div className="summary-item">
            <strong>Run ID:</strong> {evalResults.run_id}
          </div>
          <div className="summary-item">
            <strong>Models:</strong> {models.join(', ')}
          </div>
          <div className="summary-item">
            <strong>Iterations:</strong> {evalResults.iterations}
          </div>
        </div>

        <div className="model-comparisons">
          {models.map(model => {
            const modelResults = evalResults.results[model];
            const avgTime = modelResults.reduce((sum, r) => sum + r.summary.avg_time_ms, 0) / modelResults.length;
            const avgSuccess = modelResults.reduce((sum, r) => sum + r.summary.success_rate, 0) / modelResults.length;

            return (
              <div key={model} className="model-result">
                <h4>{model}</h4>
                <div className="model-metrics">
                  <div className="metric">
                    <span className="metric-label">Avg Response Time:</span>
                    <span className="metric-value">{avgTime.toFixed(0)}ms</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Success Rate:</span>
                    <span className="metric-value">{(avgSuccess * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="prompt-results">
                  {modelResults.map((promptResult, idx) => (
                    <div key={idx} className="prompt-result">
                      <div className="prompt-header">
                        <span className="prompt-text">{promptResult.prompt}</span>
                        {promptResult.tags.length > 0 && (
                          <div className="prompt-tags">
                            {promptResult.tags.map(tag => (
                              <span key={tag} className="tag">{tag}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      
                      <div className="iterations">
                        {promptResult.results.map(result => (
                          <div key={result.iteration} className={`iteration ${result.passed ? 'passed' : 'failed'}`}>
                            <span className="iteration-num">#{result.iteration}</span>
                            <span className="iteration-time">{result.time_ms}ms</span>
                            {result.error ? (
                              <XCircle size={16} />
                            ) : (
                              result.passed ? <CheckCircle size={16} /> : <AlertCircle size={16} />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="evaluation-tool">
      <div className="eval-header">
        <h2>Model Evaluation Tool</h2>
        <button 
          className="create-button"
          onClick={() => setCreateMode(!createMode)}
        >
          <Plus size={16} />
          Create Evaluation Set
        </button>
      </div>

      {error && (
        <div className="error-message">
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {createMode && (
        <div className="create-eval-set">
          <h3>Create New Evaluation Set</h3>
          
          <div className="form-group">
            <label>Name</label>
            <input
              type="text"
              value={newEvalName}
              onChange={(e) => setNewEvalName(e.target.value)}
              placeholder="e.g., Code Generation Tests"
            />
          </div>

          <div className="form-group">
            <label>Description</label>
            <textarea
              value={newEvalDescription}
              onChange={(e) => setNewEvalDescription(e.target.value)}
              placeholder="Optional description..."
              rows={2}
            />
          </div>

          <div className="prompts-section">
            <label>Test Prompts</label>
            {newEvalPrompts.map((prompt, index) => (
              <div key={index} className="prompt-item">
                <textarea
                  value={prompt.prompt}
                  onChange={(e) => updatePrompt(index, 'prompt', e.target.value)}
                  placeholder="Enter test prompt..."
                  rows={2}
                />
                <input
                  type="text"
                  value={prompt.expected_contains.join(', ')}
                  onChange={(e) => updatePrompt(index, 'expected_contains', 
                    e.target.value.split(',').map(s => s.trim()).filter(s => s)
                  )}
                  placeholder="Expected keywords (comma-separated)"
                />
                <input
                  type="text"
                  value={prompt.tags.join(', ')}
                  onChange={(e) => updatePrompt(index, 'tags',
                    e.target.value.split(',').map(s => s.trim()).filter(s => s)
                  )}
                  placeholder="Tags (comma-separated)"
                />
                {newEvalPrompts.length > 1 && (
                  <button 
                    className="remove-button"
                    onClick={() => removePrompt(index)}
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            
            <button className="add-prompt-button" onClick={addPrompt}>
              <Plus size={16} /> Add Prompt
            </button>
          </div>

          <div className="form-actions">
            <button 
              className="save-button"
              onClick={createEvalSet}
              disabled={loading}
            >
              <Save size={16} />
              {loading ? 'Creating...' : 'Create Set'}
            </button>
            <button 
              className="cancel-button"
              onClick={() => {
                setCreateMode(false);
                resetForm();
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {!createMode && (
        <div className="eval-runner">
          <div className="run-config">
            <h3>Run Evaluation</h3>
            
            <div className="config-grid">
              <div className="config-item">
                <label>Evaluation Set</label>
                <select
                  value={selectedSet?.eval_id || ''}
                  onChange={(e) => setSelectedSet(
                    evalSets.find(s => s.eval_id === e.target.value)
                  )}
                >
                  <option value="">Select evaluation set...</option>
                  {evalSets.map(set => (
                    <option key={set.eval_id} value={set.eval_id}>
                      {set.name} ({set.prompt_count} prompts)
                    </option>
                  ))}
                </select>
              </div>

              <div className="config-item">
                <label>Models</label>
                <div className="model-selection">
                  {models.map(model => (
                    <label key={model.id} className="model-checkbox">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, model.id]);
                          } else {
                            setSelectedModels(selectedModels.filter(m => m !== model.id));
                          }
                        }}
                      />
                      {model.id}
                    </label>
                  ))}
                </div>
              </div>

              <div className="config-item">
                <label>Iterations: {iterations}</label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={iterations}
                  onChange={(e) => setIterations(Number(e.target.value))}
                />
              </div>

              <div className="config-item">
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                />
              </div>

              <div className="config-item">
                <label>Max Tokens: {maxTokens}</label>
                <input
                  type="range"
                  min="100"
                  max="4000"
                  step="100"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                />
              </div>
            </div>

            <button
              className="run-button"
              onClick={runEvaluation}
              disabled={loading || !selectedSet || selectedModels.length === 0}
            >
              <Play size={16} />
              {loading ? 'Running...' : 'Run Evaluation'}
            </button>
          </div>

          {runningEval && (
            <div className="eval-progress">
              <h4>Running Evaluation...</h4>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${(runningEval.progress.completed / runningEval.progress.total) * 100}%` 
                  }}
                />
              </div>
              <div className="progress-text">
                {runningEval.progress.completed} / {runningEval.progress.total} tests completed
              </div>
            </div>
          )}

          {renderResults()}
        </div>
      )}
    </div>
  );
}

export default EvaluationTool;