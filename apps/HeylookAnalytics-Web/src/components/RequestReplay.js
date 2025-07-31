import React, { useState } from 'react';
import axios from 'axios';
import { RefreshCw, Zap } from 'lucide-react';
import './RequestReplay.css';

const API_BASE = 'http://localhost:8080';

function RequestReplay({ requestId, originalData, onClose }) {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(originalData?.model || '');
  const [temperature, setTemperature] = useState(originalData?.temperature || 0.7);
  const [maxTokens, setMaxTokens] = useState(originalData?.max_tokens || 1000);
  const [systemMessage, setSystemMessage] = useState('');
  const [useSystemMessage, setUseSystemMessage] = useState(false);
  const [replayResult, setReplayResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch available models
  React.useEffect(() => {
    fetchModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/v1/models`);
      setModels(response.data.data || []);
      if (!selectedModel && response.data.data.length > 0) {
        setSelectedModel(originalData?.model || response.data.data[0].id);
      }
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  };

  const runReplay = async () => {
    setLoading(true);
    setError(null);
    setReplayResult(null);

    try {
      const replayParams = {
        model: selectedModel,
        temperature: parseFloat(temperature),
        max_tokens: parseInt(maxTokens)
      };

      if (useSystemMessage && systemMessage) {
        replayParams.system_message = systemMessage;
      }

      const response = await axios.post(
        `${API_BASE}/v1/replay/${requestId}`,
        replayParams
      );

      setReplayResult(response.data);
    } catch (err) {
      console.error('Error replaying request:', err);
      setError(err.response?.data?.detail || 'Failed to replay request');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (ms) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const calculateDiff = (original, replay, field) => {
    if (!original || !replay) return 0;
    const diff = replay - original;
    const percentChange = (diff / original) * 100;
    return { diff, percentChange };
  };

  const renderComparison = () => {
    if (!replayResult) return null;

    const { original, replay, modifications } = replayResult;
    const timeDiff = calculateDiff(original.total_time_ms, replay.total_time_ms, 'time');
    const tokensDiff = calculateDiff(original.tokens_per_second, replay.tokens_per_second, 'tokens');

    return (
      <div className="comparison-results">
        <h3>Comparison Results</h3>
        
        {/* Modifications Summary */}
        <div className="modifications-summary">
          {modifications.model_changed && (
            <div className="modification-badge">Model Changed</div>
          )}
          {modifications.temperature_changed && (
            <div className="modification-badge">Temperature Changed</div>
          )}
          {modifications.max_tokens_changed && (
            <div className="modification-badge">Max Tokens Changed</div>
          )}
          {modifications.system_message_added && (
            <div className="modification-badge">System Message Added</div>
          )}
        </div>

        {/* Performance Comparison */}
        <div className="performance-comparison">
          <div className="comparison-metric">
            <h4>Response Time</h4>
            <div className="metric-values">
              <div className="original-value">
                <span className="label">Original:</span>
                <span className="value">{formatTime(original.total_time_ms)}</span>
              </div>
              <div className="replay-value">
                <span className="label">Replay:</span>
                <span className="value">{formatTime(replay.total_time_ms)}</span>
              </div>
              {timeDiff.diff !== 0 && (
                <div className={`diff ${timeDiff.diff < 0 ? 'improvement' : 'regression'}`}>
                  {timeDiff.diff < 0 ? '▼' : '▲'} {Math.abs(timeDiff.percentChange).toFixed(1)}%
                </div>
              )}
            </div>
          </div>

          <div className="comparison-metric">
            <h4>Tokens/Second</h4>
            <div className="metric-values">
              <div className="original-value">
                <span className="label">Original:</span>
                <span className="value">{original.tokens_per_second?.toFixed(1) || 'N/A'}</span>
              </div>
              <div className="replay-value">
                <span className="label">Replay:</span>
                <span className="value">{replay.tokens_per_second?.toFixed(1) || 'N/A'}</span>
              </div>
              {tokensDiff.diff !== 0 && (
                <div className={`diff ${tokensDiff.diff > 0 ? 'improvement' : 'regression'}`}>
                  {tokensDiff.diff > 0 ? '▲' : '▼'} {Math.abs(tokensDiff.percentChange).toFixed(1)}%
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Response Comparison */}
        <div className="response-comparison">
          <div className="response-section">
            <h4>Original Response</h4>
            <div className="response-content">
              {original.response_text || 'No response available'}
            </div>
          </div>
          
          <div className="response-section">
            <h4>Replay Response</h4>
            <div className="response-content">
              {replay.response_text || 'No response available'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="request-replay-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Request Replay</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {/* Configuration Section */}
          <div className="replay-config">
            <h3>Replay Configuration</h3>
            
            <div className="config-grid">
              <div className="config-item">
                <label>Model</label>
                <select 
                  value={selectedModel} 
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.id} {model.id === originalData?.model && '(original)'}
                    </option>
                  ))}
                </select>
              </div>

              <div className="config-item">
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
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
                  onChange={(e) => setMaxTokens(e.target.value)}
                />
              </div>
            </div>

            <div className="system-message-section">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useSystemMessage}
                  onChange={(e) => setUseSystemMessage(e.target.checked)}
                />
                Add System Message
              </label>
              
              {useSystemMessage && (
                <textarea
                  value={systemMessage}
                  onChange={(e) => setSystemMessage(e.target.value)}
                  placeholder="Enter system message..."
                  rows={3}
                />
              )}
            </div>

            <button 
              className="replay-button"
              onClick={runReplay}
              disabled={loading || !selectedModel}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="spin" />
                  Running Replay...
                </>
              ) : (
                <>
                  <Zap size={16} />
                  Run Replay
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {/* Results Section */}
          {renderComparison()}
        </div>
      </div>
    </div>
  );
}

export default RequestReplay;