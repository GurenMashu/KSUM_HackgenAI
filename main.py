import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import hashlib
from urllib.parse import urljoin, urlparse
import pandas as pd
from collections import deque, defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

@dataclass
class NewsEvent:
    """Represents a news event with metadata"""
    title: str
    url: str
    date: datetime
    summary: str
    keywords: List[str]
    relevance_score: float
    depth_level: int
    event_type: str = "general"
    
    def __hash__(self):
        return hash(self.url)

class NewsEventMapper:
    """Core class for mapping news events and their connections"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.events_cache = {}
        self.processed_urls = set()
        self.keyword_index = defaultdict(set)
        self.max_depth = 5
        self.max_events_per_level = 10
        # Load transformer model for semantic similarity
        try:
            self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading transformer model: {str(e)}")
            self.sim_model = None
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple NLP techniques"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Get word frequency and return top keywords
        word_freq = defaultdict(int)
        for word in keywords:
            word_freq[word] += 1
            
        # Return top 20 keywords by frequency
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:20]
    
    def scrape_news_content(self, url: str) -> Optional[Dict]:
        """Scrape news content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_selectors = ['h1', 'title', '.headline', '.article-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract main content
            content = ""
            content_selectors = [
                '.article-body', '.post-content', '.entry-content', 
                'article', '.story-body', 'main', '.content'
            ]
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)
                    break
            
            # If no specific content found, get all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Extract date (simplified approach)
            date = datetime.now()  # Default to current date
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'\b\w+\s+\d{1,2},\s+\d{4}\b'
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, response.text)
                if date_match:
                    try:
                        date_str = date_match.group()
                        # Try to parse the date (simplified)
                        if '-' in date_str:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                        elif '/' in date_str:
                            date = datetime.strptime(date_str, '%m/%d/%Y')
                        break
                    except:
                        continue
            
            return {
                'title': title,
                'content': content,
                'date': date,
                'url': url
            }
            
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return None

    def search_related_news(self, keywords: List[str], base_date: datetime) -> List[Dict]:
        """Fetch related news using NewsAPI (replace with your API key)"""
        API_KEY = 'df0a8efd986149f683e7d172f8f52fa2' 
        NEWS_URL = "https://newsapi.org/v2/everything"

        query = ' OR '.join(keywords)
        from_date = (base_date - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = base_date.strftime('%Y-%m-%d')

        params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'sortBy': 'relevancy',
        'language': 'en',
        'pageSize': 10,
        'apiKey': API_KEY
        }

        response = requests.get(NEWS_URL, params=params)
        articles = []

        def _calculate_relevance(article: Dict, keywords: List[str]) -> float:
            """Simple relevance scoring based on keyword frequency"""
            content = (article.get("title", "") + " " + article.get("description", "")).lower()
            count = sum(content.count(kw.lower()) for kw in keywords)
            return min(1.0, count / len(keywords))  # Cap at 1.0

        if response.status_code == 200:
            data = response.json()
            for article in data.get("articles", []):
                article_date = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                relevance = _calculate_relevance(article, keywords)
                articles.append({
                'title': article["title"],
                'url': article["url"],
                'date': article_date,
                'summary': article["description"] or "",
                'relevance': relevance
                })
        else:
            print(f"NewsAPI request failed: {response.status_code} - {response.text}")

        return sorted(articles, key=lambda x: x['relevance'], reverse=True)

    
    def calculate_relevance_score(self, event: Dict, target_keywords: List[str]) -> float:
        """Calculate semantic relevance using transformer model"""
        event_text = f"{event.get('title', '')} {event.get('summary', '')}"
        keywords_text = ' '.join(target_keywords)
        if not hasattr(self, 'sim_model') or self.sim_model is None:
            # Fallback to keyword overlap if model not loaded
            event_keywords = self.extract_keywords(event_text)
            overlap = len(set(event_keywords) & set(target_keywords))
            total_keywords = len(set(event_keywords) | set(target_keywords))
            if total_keywords == 0:
                return 0.0
            jaccard_similarity = overlap / total_keywords
            return jaccard_similarity
        try:
            # Get embeddings
            emb_event = self.sim_model.encode(event_text, convert_to_tensor=True)
            emb_keywords = self.sim_model.encode(keywords_text, convert_to_tensor=True)
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb_event, emb_keywords, dim=0).item()
            # Normalize to [0,1]
            score = max(0.0, min(1.0, (similarity + 1) / 2))
            # Boost score for recent events
            if 'date' in event:
                days_old = (datetime.now() - event['date']).days
                recency_bonus = max(0, 1 - days_old / 365)
                score = min(1.0, score * 1.2 + recency_bonus * 0.2)
            return score
        except Exception as e:
            st.error(f"Error in transformer relevance scoring: {str(e)}")
            return 0.0
    
    def build_event_graph(self, initial_url: str) -> bool:
        """Build the complete event graph starting from an initial news article"""
        try:
            # Process initial article
            initial_content = self.scrape_news_content(initial_url)
            if not initial_content:
                return False
            
            keywords = self.extract_keywords(initial_content['content'])
            initial_event = NewsEvent(
                title=initial_content['title'],
                url=initial_url,
                date=initial_content['date'],
                summary=initial_content['content'][:500] + "...",
                keywords=keywords,
                relevance_score=1.0,
                depth_level=0
            )
            
            # Add to graph
            self.graph.add_node(
                initial_url,
                event=initial_event,
                pos=(0, 0, 0),  # Add z=0 for 3D
                color='red',
                size=30
            )
            
            # Build graph using BFS
            queue = deque([(initial_event, 0)])
            processed = {initial_url}
            
            progress_bar = st.progress(0)
            total_operations = self.max_depth * self.max_events_per_level
            current_operation = 0
            
            while queue and current_operation < total_operations:
                current_event, depth = queue.popleft()
                
                if depth >= self.max_depth:
                    continue
                
                # Search for related news
                related_news = self.search_related_news(current_event.keywords, current_event.date)
                
                # Process related articles
                added_count = 0
                for news in related_news[:self.max_events_per_level]:
                    if news['url'] in processed or added_count >= 5:
                        continue
                    
                    relevance = self.calculate_relevance_score(news, current_event.keywords)
                    if relevance < 0.3:  # Relevance threshold
                        continue
                    
                    related_event = NewsEvent(
                        title=news['title'],
                        url=news['url'],
                        date=news['date'],
                        summary=news['summary'],
                        keywords=self.extract_keywords(news['summary']),
                        relevance_score=relevance,
                        depth_level=depth + 1
                    )
                    
                    # Add to graph with 3D positioning
                    angle = (added_count * 360 / 8) * np.pi / 180
                    radius = (depth + 1) * 2
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    # z coordinate: spiral or layer by depth
                    z = depth * 2 + np.sin(angle) * 1.5

                    color = 'orange' if depth == 0 else 'lightblue' if depth < 3 else 'lightgreen'
                    size = max(10, 25 - depth * 3)

                    self.graph.add_node(
                        news['url'],
                        event=related_event,
                        pos=(x, y, z),
                        color=color,
                        size=size
                    )
                    
                    # Add edge with weight based on relevance
                    self.graph.add_edge(
                        current_event.url,
                        news['url'],
                        weight=relevance,
                        relationship="leads_to" if news['date'] > current_event.date else "caused_by"
                    )
                    
                    queue.append((related_event, depth + 1))
                    processed.add(news['url'])
                    added_count += 1
                    current_operation += 1
                    
                    progress_bar.progress(min(1.0, current_operation / total_operations))
                    time.sleep(0.1)  # Small delay for visualization
            
            progress_bar.empty()
            return True
            
        except Exception as e:
            st.error(f"Error building event graph: {str(e)}")
            return False
    
    def create_interactive_plot(self) -> go.Figure:
        """Create an interactive Plotly graph visualization"""
        if not self.graph.nodes():
            return go.Figure()
        
        # Extract node information for 3D
        node_x, node_y, node_z, node_text, node_size, node_color, node_custom = [], [], [], [], [], [], []
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            event = self.graph.nodes[node]['event']
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])
            # Node label is just the title
            node_text.append(event.title[:50] + "..." if len(event.title) > 50 else event.title)
            node_size.append(self.graph.nodes[node]['size'])
            node_color.append(event.relevance_score)
            # Add relevance, date, and url to customdata for popup
            node_custom.append([event.relevance_score, event.date.strftime('%Y-%m-%d'), event.url])

        # Extract edge information for 3D
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.graph.edges():
            x0, y0, z0 = self.graph.nodes[edge[0]]['pos']
            x1, y1, z1 = self.graph.nodes[edge[1]]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            text=node_text,
            mode='markers+text',
            hovertemplate='<b>%{text}</b><br>Relevance: %{customdata[0]:.2f}<br>Date: %{customdata[1]}<br><a href="%{customdata[2]}" target="_blank">Visit Article</a><extra></extra>',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Relevance Score")
            ),
            customdata=node_custom,
            textposition="middle center"
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                            text='News Event Mind Map (3D)',
                            font=dict(size=16)  
                            ),
                           showlegend=False,
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click on nodes to view article details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           scene=dict(
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           ),
                           height=700
                       ))
        return fig
    
    def get_event_details(self, url: str) -> Optional[NewsEvent]:
        """Get detailed information about a specific event"""
        if url in self.graph.nodes():
            return self.graph.nodes[url]['event']
        return None

def main():
    st.set_page_config(
        page_title="News Event Mind Map Generator",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("X")
    st.markdown("""
    **Discover the deep connections behind any news event**
    
    This tool creates an interactive mind map showing the chain of events, causes, and connections 
    that led to a specific news story. Unlike simple summaries, this provides a comprehensive 
    visual timeline of related events and their relationships.
    """)
    
    # Initialize session state
    if 'mapper' not in st.session_state:
        st.session_state.mapper = NewsEventMapper()
    
    if 'graph_built' not in st.session_state:
        st.session_state.graph_built = False
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    max_depth = st.sidebar.slider("Analysis Depth", 1, 10, 5, 
                                  help="How many levels deep to analyze connections")
    max_events = st.sidebar.slider("Events per Level", 3, 20, 10,
                                   help="Maximum events to analyze at each level")
    
    st.session_state.mapper.max_depth = max_depth
    st.session_state.mapper.max_events_per_level = max_events
    
    # Main input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        news_url = st.text_input(
            "Enter a news article URL:",
            placeholder="https://example.com/news-article",
            help="Paste the URL of any news article to analyze its background and connections"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    # Sample URLs for testing
    with st.expander("📝 Try Sample URLs"):
        sample_urls = [
            "https://example.com/sample-political-news",
            "https://example.com/sample-economic-news", 
            "https://example.com/sample-technology-news"
        ]
        
        for i, url in enumerate(sample_urls):
            if st.button(f"Sample {i+1}: {url}", key=f"sample_{i}"):
                st.session_state.sample_url = url
                news_url = url
    
    # Process analysis
    if analyze_button and news_url:
        with st.spinner("🔍 Analyzing news article and building connection map..."):
            st.info("📡 Scraping article content...")
            
            success = st.session_state.mapper.build_event_graph(news_url)
            
            if success:
                st.session_state.graph_built = True
                st.success("✅ Analysis complete! Explore the mind map below.")
            else:
                st.error("❌ Failed to analyze the article. Please check the URL and try again.")
    
    # Display results
    if st.session_state.graph_built:
        st.header("📊 Interactive Mind Map")
        
        # Create and display the plot
        fig = st.session_state.mapper.create_interactive_plot()
        
        if fig.data:
            selected_points = st.plotly_chart(
                fig, 
                use_container_width=True,
                key="mindmap_plot"
            )
            
            # Event details section
            st.header("📋 Event Details")
            
            # Show graph statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Events", len(st.session_state.mapper.graph.nodes()))
            
            with col2:
                st.metric("Connections", len(st.session_state.mapper.graph.edges()))
            
            with col3:
                avg_relevance = np.mean([
                    st.session_state.mapper.graph.nodes[node]['event'].relevance_score 
                    for node in st.session_state.mapper.graph.nodes()
                ])
                st.metric("Avg Relevance", f"{avg_relevance:.2f}")
            
            with col4:
                max_depth = max([
                    st.session_state.mapper.graph.nodes[node]['event'].depth_level 
                    for node in st.session_state.mapper.graph.nodes()
                ])
                st.metric("Analysis Depth", max_depth + 1)
            
            # Event list
            st.subheader("🔗 All Connected Events")
            
            events_data = []
            for node in st.session_state.mapper.graph.nodes():
                event = st.session_state.mapper.graph.nodes[node]['event']
                events_data.append({
                    'Title': event.title,
                    'Date': event.date.strftime('%Y-%m-%d'),
                    'Relevance': f"{event.relevance_score:.2f}",
                    'Depth': event.depth_level,
                    'URL': event.url
                })
            
            df = pd.DataFrame(events_data)
            df = df.sort_values(['Depth', 'Relevance'], ascending=[True, False])
            
            # Make URLs clickable
            df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">View Article</a>')
            
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
            
            # Download options
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Graph Data (JSON)"):
                    graph_data = {
                        'nodes': [
                            {
                                'id': node,
                                'title': st.session_state.mapper.graph.nodes[node]['event'].title,
                                'date': st.session_state.mapper.graph.nodes[node]['event'].date.isoformat(),
                                'relevance': st.session_state.mapper.graph.nodes[node]['event'].relevance_score,
                                'depth': st.session_state.mapper.graph.nodes[node]['event'].depth_level,
                                'url': st.session_state.mapper.graph.nodes[node]['event'].url
                            }
                            for node in st.session_state.mapper.graph.nodes()
                        ],
                        'edges': [
                            {
                                'source': edge[0],
                                'target': edge[1],
                                'weight': st.session_state.mapper.graph.edges[edge].get('weight', 1.0),
                                'relationship': st.session_state.mapper.graph.edges[edge].get('relationship', 'related')
                            }
                            for edge in st.session_state.mapper.graph.edges()
                        ]
                    }
                    
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(graph_data, indent=2),
                        file_name=f"news_mindmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📄 Download Event List (CSV)"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"news_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("No data available for visualization.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 How it works:**
    1. **Input Analysis**: Scrapes and analyzes the initial news article
    2. **Keyword Extraction**: Identifies key topics and entities  
    3. **Connection Discovery**: Searches for related events and news
    4. **Relevance Scoring**: Calculates how events connect to each other
    5. **Graph Building**: Creates an interactive network of related events
    6. **Timeline Mapping**: Shows the chronological flow of events
    
    **🔧 Technical Features:**
    - Deep graph analysis beyond LLM limitations
    - Interactive visualization with Plotly
    - Relevance-based event scoring
    - Chronological event mapping
    - Exportable data formats
    """)

if __name__ == "__main__":
    main()