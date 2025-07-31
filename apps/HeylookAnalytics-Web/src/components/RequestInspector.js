import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertCircle, Copy, Download, ChevronDown, ChevronRight, RefreshCw } from 'lucide-react';
import RequestReplay from './RequestReplay';
import './RequestInspector.css';

const API_BASE = 'http://localhost:8080';

function RequestInspector() {
  const [recentRequests, setRecentRequests] = useState([]);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [requestDetails, setRequestDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    request: true,
    response: true,
    timing: true,
    tokens: true,
    resources: false
  });
  const [showReplay, setShowReplay] = useState(false);

  // Fetch recent requests
  useEffect(() => {
    fetchRecentRequests();
  }, []);

  const fetchRecentRequests = async () => {
    try {
      const response = await axios.post(`${API_BASE}/v1/data/query`, {
        query: `
          SELECT request_id, timestamp, model, request_type, 
                 total_time_ms, success, error_type,
                 prompt_tokens + completion_tokens as total_tokens
          FROM request_logs 
          ORDER BY timestamp DESC 
          LIMIT 50
        `
      });
      
      if (response.data.data) {
        setRecentRequests(response.data.data.map(row => ({
          request_id: row[0],
          timestamp: row[1],
          model: row[2],
          request_type: row[3],
          total_time_ms: row[4],
          success: row[5],
          error_type: row[6],
          total_tokens: row[7]
        })));
      }
    } catch (err) {
      console.error('Error fetching recent requests:', err);
      setError('Failed to fetch recent requests');
    }
  };

  const fetchRequestDetails = async (requestId) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE}/v1/data/request/${requestId}`);
      setRequestDetails(response.data);
      setSelectedRequest(requestId);
    } catch (err) {
      console.error('Error fetching request details:', err);
      setError(err.response?.data?.detail || 'Failed to fetch request details');
      setRequestDetails(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const copyToClipboard = (text, format = 'json') => {
    const content = format === 'json' ? JSON.stringify(text, null, 2) : text;
    navigator.clipboard.writeText(content);
    // TODO: Add toast notification
  };

  const exportAsCurl = () => {
    if (!requestDetails) return;
    
    const messages = requestDetails.messages || [];
    const curlCommand = `curl -X POST ${API_BASE}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${requestDetails.model}",
    "messages": ${JSON.stringify(messages, null, 2).split('\n').join('\n    ')},
    "temperature": ${requestDetails.temperature || 0.7},
    "max_tokens": ${requestDetails.max_tokens || 1000}
  }'`;
    
    copyToClipboard(curlCommand, 'curl');
  };

  const exportAsPython = () => {
    if (!requestDetails) return;
    
    const messages = requestDetails.messages || [];
    const pythonCode = `from openai import OpenAI

client = OpenAI(
    base_url="${API_BASE}/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="${requestDetails.model}",
    messages=${JSON.stringify(messages, null, 2).split('\n').join('\n    ')},
    temperature=${requestDetails.temperature || 0.7},
    max_tokens=${requestDetails.max_tokens || 1000}
)

print(response.choices[0].message.content)`;
    
    copyToClipboard(pythonCode, 'python');
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const renderImageContent = (content) => {
    if (!content || typeof content !== 'object') return null;
    
    if (content.type === 'image_url' && content.image_url?.url) {
      const url = content.image_url.url;
      if (url.startsWith('data:image')) {
        return (
          <div className="image-preview">
            <img src={url} alt="User uploaded" />
          </div>
        );
      }
    }
    return null;
  };

  const renderMessage = (message, index) => {
    const isMultipart = Array.isArray(message.content);
    
    return (
      <div key={index} className={`message ${message.role}`}>
        <div className="message-role">{message.role}</div>
        <div className="message-content">
          {isMultipart ? (
            message.content.map((part, partIndex) => (
              <div key={partIndex}>
                {part.type === 'text' && <div className="text-content">{part.text}</div>}
                {part.type === 'image_url' && renderImageContent(part)}
              </div>
            ))
          ) : (
            <div className="text-content">{message.content}</div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="request-inspector">
      <div className="inspector-layout">
        {/* Request List */}
        <div className="request-list">
          <h3>Recent Requests</h3>
          <div className="request-items">
            {recentRequests.map(request => (
              <div
                key={request.request_id}
                className={`request-item ${selectedRequest === request.request_id ? 'selected' : ''} ${!request.success ? 'error' : ''}`}
                onClick={() => fetchRequestDetails(request.request_id)}
              >
                <div className="request-time">{formatTimestamp(request.timestamp)}</div>
                <div className="request-model">{request.model}</div>
                <div className="request-stats">
                  <span className="duration">{formatDuration(request.total_time_ms)}</span>
                  {request.total_tokens && <span className="tokens">{request.total_tokens} tokens</span>}
                  {!request.success && <span className="error-badge">Error</span>}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Request Details */}
        <div className="request-details">
          {loading && <div className="loading">Loading request details...</div>}
          
          {error && (
            <div className="error-message">
              <AlertCircle size={20} />
              {error}
            </div>
          )}
          
          {requestDetails && !loading && (
            <>
              {/* Header */}
              <div className="details-header">
                <h3>Request Details</h3>
                <div className="header-actions">
                  <button onClick={() => setShowReplay(true)} title="Replay Request">
                    <RefreshCw size={16} /> Replay
                  </button>
                  <button onClick={() => copyToClipboard(requestDetails)} title="Copy JSON">
                    <Copy size={16} /> JSON
                  </button>
                  <button onClick={exportAsCurl} title="Copy as cURL">
                    <Download size={16} /> cURL
                  </button>
                  <button onClick={exportAsPython} title="Copy as Python">
                    <Download size={16} /> Python
                  </button>
                </div>
              </div>

              {/* Metadata */}
              <div className="request-metadata">
                <div className="metadata-item">
                  <strong>Request ID:</strong> {requestDetails.request_id}
                </div>
                <div className="metadata-item">
                  <strong>Timestamp:</strong> {formatTimestamp(requestDetails.timestamp)}
                </div>
                <div className="metadata-item">
                  <strong>Model:</strong> {requestDetails.model}
                </div>
                <div className="metadata-item">
                  <strong>Provider:</strong> {requestDetails.provider}
                </div>
              </div>

              {/* Request Section */}
              <div className="inspector-section">
                <div className="section-header" onClick={() => toggleSection('request')}>
                  {expandedSections.request ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  <h4>Request</h4>
                </div>
                {expandedSections.request && (
                  <div className="section-content">
                    <div className="messages">
                      {requestDetails.messages?.map((msg, idx) => renderMessage(msg, idx))}
                    </div>
                    <div className="request-params">
                      <div className="param">
                        <strong>Temperature:</strong> {requestDetails.temperature}
                      </div>
                      <div className="param">
                        <strong>Max Tokens:</strong> {requestDetails.max_tokens}
                      </div>
                      {requestDetails.num_images > 0 && (
                        <div className="param">
                          <strong>Images:</strong> {requestDetails.num_images}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Response Section */}
              <div className="inspector-section">
                <div className="section-header" onClick={() => toggleSection('response')}>
                  {expandedSections.response ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  <h4>Response</h4>
                </div>
                {expandedSections.response && (
                  <div className="section-content">
                    {requestDetails.status.success ? (
                      <div className="response-text">
                        {requestDetails.response_text || 'No response text available'}
                      </div>
                    ) : (
                      <div className="error-details">
                        <div className="error-type">Error: {requestDetails.status.error_type}</div>
                        <div className="error-message">{requestDetails.status.error_message}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Timing Section */}
              <div className="inspector-section">
                <div className="section-header" onClick={() => toggleSection('timing')}>
                  {expandedSections.timing ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  <h4>Timing Breakdown</h4>
                </div>
                {expandedSections.timing && (
                  <div className="section-content">
                    <div className="timing-chart">
                      {Object.entries(requestDetails.timing).map(([key, value]) => {
                        if (!value || key === 'total_ms') return null;
                        const percentage = (value / requestDetails.timing.total_ms) * 100;
                        return (
                          <div key={key} className="timing-item">
                            <div className="timing-label">
                              {key.replace(/_/g, ' ').replace('ms', '')}
                            </div>
                            <div className="timing-bar">
                              <div 
                                className="timing-fill" 
                                style={{ width: `${percentage}%` }}
                              />
                              <span className="timing-value">{formatDuration(value)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="timing-total">
                      <strong>Total Time:</strong> {formatDuration(requestDetails.timing.total_ms)}
                    </div>
                  </div>
                )}
              </div>

              {/* Tokens Section */}
              <div className="inspector-section">
                <div className="section-header" onClick={() => toggleSection('tokens')}>
                  {expandedSections.tokens ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  <h4>Token Usage</h4>
                </div>
                {expandedSections.tokens && (
                  <div className="section-content">
                    <div className="token-stats">
                      <div className="token-item">
                        <div className="token-label">Prompt Tokens</div>
                        <div className="token-value">{requestDetails.tokens.prompt || 0}</div>
                      </div>
                      <div className="token-item">
                        <div className="token-label">Completion Tokens</div>
                        <div className="token-value">{requestDetails.tokens.completion || 0}</div>
                      </div>
                      <div className="token-item">
                        <div className="token-label">Total Tokens</div>
                        <div className="token-value">{requestDetails.tokens.total || 0}</div>
                      </div>
                      <div className="token-item">
                        <div className="token-label">Tokens/Second</div>
                        <div className="token-value">
                          {requestDetails.tokens.per_second?.toFixed(1) || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Resources Section */}
              {(requestDetails.resources.memory_gb || requestDetails.resources.gpu_utilization) && (
                <div className="inspector-section">
                  <div className="section-header" onClick={() => toggleSection('resources')}>
                    {expandedSections.resources ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    <h4>Resource Usage</h4>
                  </div>
                  {expandedSections.resources && (
                    <div className="section-content">
                      <div className="resource-stats">
                        {requestDetails.resources.memory_gb && (
                          <div className="resource-item">
                            <div className="resource-label">Memory Used</div>
                            <div className="resource-value">
                              {requestDetails.resources.memory_gb.toFixed(2)} GB
                            </div>
                          </div>
                        )}
                        {requestDetails.resources.gpu_utilization && (
                          <div className="resource-item">
                            <div className="resource-label">GPU Utilization</div>
                            <div className="resource-value">
                              {requestDetails.resources.gpu_utilization.toFixed(1)}%
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
      
      {/* Request Replay Modal */}
      {showReplay && requestDetails && (
        <RequestReplay
          requestId={selectedRequest}
          originalData={requestDetails}
          onClose={() => setShowReplay(false)}
        />
      )}
    </div>
  );
}

export default RequestInspector;