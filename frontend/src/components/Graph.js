import React, { useState, useEffect, useRef } from 'react';
import Select from 'react-select';
import * as d3 from 'd3';
import { api } from '../services/api';
import '../styles/components/Graph.css';

export function GraphTab({ contexts }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [selectedContext, setSelectedContext] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [filteredGraphData, setFilteredGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeCount, setNodeCount] = useState(50); // Start with 50 nodes
  const svgRef = useRef(null);
  
  // Filter out contexts containing "cells" and prepare options for react-select
  const contextOptions = contexts
    .filter(ctx => !ctx.toLowerCase().includes('cells'))
    .map(ctx => ({ value: ctx, label: ctx }));
  
  const handleContextChange = async (selected) => {
    setSelectedContext(selected);
    setGraphData(null); // Clear previous graph
    setFilteredGraphData(null);
    setSelectedNode(null); // Clear selected node
    setNodeCount(50); // Reset to 50 nodes
    
    if (selected && selected.value) {
      setLoading(true);
      try {
        // Fetch the PPI graph data - already sorted by backend!
        const data = await api.fetchPPIGraph(selected.value);
        
        // No need to calculate degrees or sort - backend already did it
        setGraphData({
          sortedNodes: data.nodes,  // Already sorted by degree
          edges: data.edges,
          originalNodes: data.nodes
        });
        
        // Initially show top 50 nodes
        filterGraphByNodeCount({
          sortedNodes: data.nodes,
          edges: data.edges
        }, 50);
      } catch (error) {
        console.error('Error fetching graph:', error);
        alert('Failed to load graph data');
      } finally {
        setLoading(false);
      }
    }
  };
  
  const filterGraphByNodeCount = (data, count) => {
    if (!data) return;
    
    // Take top N nodes by degree - already sorted!
    const topNodes = data.sortedNodes.slice(0, count);
    const nodeIds = new Set(topNodes.map(n => n.id));
    
    // Filter edges to only include those between selected nodes
    const filteredEdges = data.edges.filter(edge => {
      // Check if both source and target are in our selected nodes
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });
    
    console.log(`Filtered to ${topNodes.length} nodes and ${filteredEdges.length} edges`);
    
    setFilteredGraphData({
      nodes: topNodes,
      edges: filteredEdges
    });
  };
  
  const handleNodeCountChange = (e) => {
    const newCount = parseInt(e.target.value);
    if (!isNaN(newCount) && newCount >= 10 && newCount <= graphData.sortedNodes.length) {
      setNodeCount(newCount);
      filterGraphByNodeCount(graphData, newCount);
    }
  };
  
  const handleNodeCountInputChange = (e) => {
    const value = e.target.value;
    if (value === '') {
      setNodeCount('');
    } else {
      const newCount = parseInt(value);
      if (!isNaN(newCount) && newCount >= 10 && newCount <= graphData.sortedNodes.length) {
        setNodeCount(newCount);
        filterGraphByNodeCount(graphData, newCount);
      }
    }
  };

  useEffect(() => {
    if (filteredGraphData && svgRef.current) {
      drawGraph();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredGraphData]);

  const drawGraph = () => {
    // Clear previous graph
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = 800;
    const height = 600;
    
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    // Create a group for zoom/pan
    const g = svg.append("g");
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });
    
    svg.call(zoom);
    
    // Add zoom controls
    const zoomIn = () => {
      svg.transition().call(zoom.scaleBy, 1.3);
    };
    
    const zoomOut = () => {
      svg.transition().call(zoom.scaleBy, 0.7);
    };
    
    const resetZoom = () => {
      svg.transition().call(zoom.transform, d3.zoomIdentity);
    };
    
    // Store zoom functions for buttons
    window.graphZoomIn = zoomIn;
    window.graphZoomOut = zoomOut;
    window.graphResetZoom = resetZoom;
    
    // Create force simulation with more compact layout
    const simulation = d3.forceSimulation(filteredGraphData.nodes)
      .force("link", d3.forceLink(filteredGraphData.edges)
        .id(d => d.id)
        .distance(30)) // Slightly longer for better visibility
      .force("charge", d3.forceManyBody()
        .strength(-150) // Slightly more repulsion
        .distanceMax(100))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => 5 + Math.sqrt(d.degree) * 2));
    
    // Create container groups in correct order
    const linkGroup = g.append("g").attr("class", "links");
    const nodeGroup = g.append("g").attr("class", "nodes");
    
    // Create edges FIRST (so they appear behind nodes)
    const link = linkGroup
      .selectAll("line")
      .data(filteredGraphData.edges)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", 0.5);
    
    // Create nodes AFTER edges
    const node = nodeGroup
      .selectAll("g")
      .data(filteredGraphData.nodes)
      .enter().append("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
    
    // Scale node size based on degree (smaller scale for more compact view)
    const sizeScale = d3.scaleLinear()
      .domain([0, d3.max(filteredGraphData.nodes, d => d.degree)])
      .range([3, 15]); // Smaller nodes overall
    
    // Add circles for nodes
    node.append("circle")
      .attr("r", d => sizeScale(d.degree))
      .attr("fill", d => {
        // Color by degree (hub vs peripheral)
        const maxDegree = d3.max(filteredGraphData.nodes, n => n.degree);
        if (d.degree > maxDegree * 0.7) return "#ff6b6b"; // High degree - hub
        if (d.degree < maxDegree * 0.3) return "#4ecdc4"; // Low degree - peripheral
        return "#69b3a2"; // Medium degree
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        // Double-click to recenter
        if (event.detail === 2) {
          highlightNode(d);
        }
      })
      .on("mouseover", function(event, d) {
        d3.select(this).attr("stroke-width", 4);
        // Show tooltip
        const tooltip = d3.select("body").append("div")
          .attr("class", "graph-tooltip")
          .style("position", "absolute")
          .style("padding", "10px")
          .style("background", "rgba(0,0,0,0.8)")
          .style("color", "white")
          .style("border-radius", "5px")
          .style("pointer-events", "none")
          .style("opacity", 0);
        
        tooltip.transition()
          .duration(200)
          .style("opacity", .9);
        
        tooltip.html(`<strong>${d.name || d.id}</strong><br/>
                     Degree: ${d.degree || 0}<br/>
                     Type: ${d.type || 'protein'}`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function(event, d) {
        d3.select(this).attr("stroke-width", 2);
        d3.selectAll(".graph-tooltip").remove();
      });
    
    // Add labels (only for high-degree nodes to reduce clutter)
    const labelThreshold = d3.quantile(filteredGraphData.nodes.map(d => d.degree), 0.8);
    node.filter(d => d.degree >= labelThreshold)
      .append("text")
      .text(d => d.name || d.id)
      .attr("x", 0)
      .attr("y", d => -sizeScale(d.degree) - 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "#333");
    
    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });
    
    // Stop simulation after a certain time to improve performance
    setTimeout(() => simulation.stop(), 3000);
    
    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };
  
  // Handle protein search
  const handleSearch = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    
    if (value.trim() && graphData) {
      // Search in ALL nodes, not just filtered ones
      const results = graphData.sortedNodes.filter(node => 
        node.id.toLowerCase().includes(value.toLowerCase()) ||
        (node.name && node.name.toLowerCase().includes(value.toLowerCase()))
      );
      setSearchResults(results.slice(0, 10)); // Limit to 10 results
      setShowSearchResults(true);
    } else {
      setSearchResults([]);
      setShowSearchResults(false);
    }
  };

  // Handle Enter key press
  const handleSearchKeyPress = (e) => {
    if (e.key === 'Enter' && searchResults.length > 0) {
      // Select the first result
      selectProtein(searchResults[0]);
    }
  };

  const selectProtein = (node) => {
    setSearchTerm(node.name || node.id);
    setShowSearchResults(false);
    
    // Always show neighborhood view for selected protein
    showProteinNeighborhood(node);
  };
  
  const showProteinNeighborhood = (centerNode) => {
    // Show loading message
    const message = document.createElement('div');
    message.className = 'graph-message';
    message.innerHTML = `Loading <strong>${centerNode.name || centerNode.id}</strong> neighborhood...`;
    
    const graphContainer = document.querySelector('.graph-container');
    if (graphContainer) {
      const existingMessage = graphContainer.querySelector('.graph-message');
      if (existingMessage) existingMessage.remove();
      graphContainer.appendChild(message);
    }
    
    // Get all edges connected to this node
    const connectedEdges = graphData.edges.filter(edge => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      return sourceId === centerNode.id || targetId === centerNode.id;
    });
    
    // Get all connected node IDs
    const neighborIds = new Set([centerNode.id]);
    connectedEdges.forEach(edge => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      neighborIds.add(sourceId);
      neighborIds.add(targetId);
    });
    
    // Get the actual node objects
    const neighborNodes = graphData.sortedNodes.filter(n => neighborIds.has(n.id));
    
    // Update the filtered data with just this neighborhood
    setFilteredGraphData({
      nodes: neighborNodes,
      edges: connectedEdges
    });
    
    // Update message
    if (message) {
      message.innerHTML = `
        Showing <strong>${centerNode.name || centerNode.id}</strong> and ${neighborNodes.length - 1} neighbors
        <button class="recenter-btn" onclick="window.resetToFullView()">Back to top ${nodeCount} view</button>
      `;
      
      // Add reset function
      window.resetToFullView = () => {
        filterGraphByNodeCount(graphData, nodeCount);
        message.remove();
      };
      
      setTimeout(() => message.remove(), 10000);
    }
    
    // Highlight the searched node after graph redraws
    setTimeout(() => {
      highlightNode(centerNode);
    }, 500);
  };
  
  const highlightNode = (node) => {
    // Get the SVG and its dimensions
    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 600;
    
    // Find the node in the graph
    const graphNode = d3.selectAll("circle")
      .filter(d => d.id === node.id);
    
    if (!graphNode.empty()) {
      // Highlight: dim all nodes except the selected one
      d3.selectAll("circle")
        .transition()
        .duration(300)
        .style("opacity", 0.2)
        .attr("stroke-width", 2);
      
      // Highlight the selected node
      graphNode
        .transition()
        .duration(300)
        .style("opacity", 1)
        .attr("stroke-width", 6)
        .attr("stroke", "#8b5cf6");
      
      // Also dim the edges
      d3.selectAll("line")
        .transition()
        .duration(300)
        .style("opacity", 0.1);
      
      // Highlight edges connected to this node
      d3.selectAll("line")
        .filter(d => d.source.id === node.id || d.target.id === node.id)
        .transition()
        .duration(300)
        .style("opacity", 0.6)
        .attr("stroke", "#8b5cf6")
        .attr("stroke-width", 1);
      
      // Get the node's position
      const nodeData = graphNode.datum();
      if (nodeData && nodeData.x && nodeData.y) {
        // Calculate the transform to center the node
        const zoom = d3.zoom().on("zoom", (event) => {
          svg.select("g").attr("transform", event.transform);
        });
        
        // Zoom and center on the node
        svg.transition()
          .duration(750)
          .call(
            zoom.transform,
            d3.zoomIdentity
              .translate(width / 2, height / 2)
              .scale(2)
              .translate(-nodeData.x, -nodeData.y)
          );
      }
    }
  };

  const clearSearch = () => {
    setSearchTerm('');
    setSearchResults([]);
    setShowSearchResults(false);
    
    // Reset all visual changes
    d3.selectAll("circle")
      .transition()
      .duration(300)
      .style("opacity", 1)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
    
    d3.selectAll("line")
      .transition()
      .duration(300)
      .style("opacity", 0.4)
      .attr("stroke", "#999")
      .attr("stroke-width", 0.5);
    
    // Reset zoom
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom().on("zoom", (event) => {
      svg.select("g").attr("transform", event.transform);
    });
    
    svg.transition()
      .duration(750)
      .call(zoom.transform, d3.zoomIdentity);
  };
  
  return (
    <div className="tab-content graph-explorer">
      <h2>Explore ALZ-GNN: Inputs & Embeddings</h2>
      
      <div className="graph-controls">
        <div className="context-selector">
          <label>Select Context:</label>
          <Select
            name="context"
            options={contextOptions}
            value={selectedContext}
            onChange={handleContextChange}
            className="context-select"
            classNamePrefix="select"
            placeholder="Choose a context..."
            isClearable
          />
        </div>
        
        {graphData && (
          <div className="graph-info">
            <span>Total Nodes: {graphData.sortedNodes.length}</span>
            <span>Total Edges: {graphData.edges.length}</span>
            <span>Displaying: {filteredGraphData?.nodes.length || 0} nodes</span>
          </div>
        )}
        
        {filteredGraphData && (
          <div className="protein-search">
            <div className="graph-search-wrapper">
              <input
                type="text"
                id="protein-search"
                name="protein-search"
                className="search-input"
                placeholder="Search proteins..."
                value={searchTerm}
                onChange={handleSearch}
                onKeyPress={handleSearchKeyPress}
                aria-label="Search proteins"
                autoComplete="off"
              />
              {searchTerm && (
                <button className="clear-search" onClick={clearSearch}>×</button>
              )}
              {showSearchResults && searchResults.length > 0 && (
                <div className="search-results">
                  {searchResults.map(node => (
                    <div 
                      key={node.id} 
                      className="search-result-item"
                      onClick={() => selectProtein(node)}
                    >
                      <span>{node.name || node.id}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
        
        {graphData && (
          <div className="node-slider">
            <label htmlFor="nodeCount">
              Number of nodes to display:
            </label>
            <input
              id="nodeCount"
              type="range"
              min="10"
              max={graphData.sortedNodes.length}
              value={nodeCount}
              onChange={handleNodeCountChange}
              step="10"
              className="node-count-range"
            />
            <input
              type="number"
              min="10"
              max={graphData.sortedNodes.length}
              value={nodeCount}
              onChange={handleNodeCountInputChange}
              className="node-count-input"
              placeholder="Nodes"
            />
            <span className="slider-info">
              (Max: {graphData.sortedNodes.length.toLocaleString()})
            </span>
          </div>
        )}
      </div>
      
      {loading && <div className="loading">Loading graph data...</div>}
      
      {filteredGraphData && !loading && (
        <div className="graph-container">
          <div className="zoom-controls">
            <button onClick={() => window.graphZoomIn && window.graphZoomIn()}>+</button>
            <button onClick={() => window.graphZoomOut && window.graphZoomOut()}>-</button>
            <button onClick={() => window.graphResetZoom && window.graphResetZoom()}>Reset</button>
          </div>
          
          <svg ref={svgRef} className="graph-svg"></svg>
          
          <div className="graph-legend">
            <h4>Node Size & Color</h4>
            <small>
              • Size = degree (connections)<br/>
              • <span className="legend-hub">Red</span> = Hub nodes<br/>
              • <span className="legend-medium">Green</span> = Medium<br/>
              • <span className="legend-peripheral">Teal</span> = Peripheral<br/>
              <strong>Controls:</strong>
              • Scroll to zoom<br/>
              • Click and drag to pan<br/>
              • Drag nodes to reposition
            </small>
          </div>
          
          {selectedNode && (
            <div className="node-details">
              <h3>Node Details</h3>
              <button 
                className="close-btn"
                onClick={() => setSelectedNode(null)}
              >×</button>
              <p><strong>Name:</strong> {selectedNode.name || selectedNode.id}</p>
              <p><strong>Degree:</strong> {selectedNode.degree || 0}</p>
              <p><strong>Rank:</strong> #{graphData.sortedNodes.findIndex(n => n.id === selectedNode.id) + 1} most connected</p>
              <p><strong>5 Closest Proteins:</strong> { "this will be computed using cosine similarity"}</p><button 
                className="focus-btn"
                onClick={() => showProteinNeighborhood(selectedNode)}
              >
                Show Neighborhood
              </button>
              {selectedNode.annotations && (
                <div>
                  <strong>Annotations:</strong>
                  <ul>
                    {Object.entries(selectedNode.annotations).map(([key, value]) => (
                      <li key={key}>{key}: {value}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {!selectedContext && !loading && (
  <div className="empty-state">
    <div className="empty-state-content">
      <h3>🧬 Protein Explorer</h3>
      
      <div className="getting-started">
        <strong>Getting Started:</strong>
        <ol>
          <li>Choose a context to visulaize the corresonding PPI used to train ALZ-GNN</li>
          <li>Use the slider to control how many top proteins to display, initial display shows top connected proteins</li>
          <li>Search for specific proteins or click any node to explore</li>
          
        </ol>
      </div>
      
      
      <div className="features-grid">
        <div className="feature-card">
          <span className="feature-icon">🔍</span>
          <h4>Search Capability</h4>
          <p>Once the graph is rendered, type any protein by name to instantly view its local neighborhood and connections</p>
        </div>
        
        <div className="feature-card">
          <span className="feature-icon">🎯</span>
          <h4>Interactive Exploration</h4>
          <p>Click nodes to see details, view closest proteins by embedding similarity, and explore embedding space</p>
        </div>
        
      </div>
    </div>
  </div>
)}
    </div>
  );
}