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
from PromptEngine import KeywordEngine
import gc

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
        API_KEY = '5b9b534d991d42f69b66a9c74eb281b9' 
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
                    'title': article.get("title") or "",
                    'url': article.get("url") or "",
                    'date': article_date,
                    'summary': article.get("description") or "",
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
            # Get embeddings with memory management
            with torch.no_grad():  # Prevent gradient computation
                emb_event = self.sim_model.encode(event_text, convert_to_tensor=True)
                emb_keywords = self.sim_model.encode(keywords_text, convert_to_tensor=True)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(emb_event, emb_keywords, dim=0).item()
                
                # Clean up tensors
                del emb_event, emb_keywords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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

    
    def build_event_graph_from_prompt(self, user_prompt: str, model, tokenizer) -> bool:
        """Build the event graph starting from a user prompt using KeywordEngine"""
        try:
            # Memory cleanup at start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            keyword_engine = KeywordEngine(model=model, tokenizer=tokenizer)
            extracted = keyword_engine.extract_keyword(user_prompt)
            
            # Cleanup after keyword extraction
            del keyword_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Parse keywords from output (maintain original parsing logic)
            keywords = []
            try:
                parsed = json.loads(extracted)
                for v in parsed.values():
                    if isinstance(v, list):
                        keywords.extend(v)
            except Exception:
                lines = extracted.splitlines()
                for line in lines:
                    m = re.match(r"Keywords \d+: (.+)", line)
                    if m:
                        keywords += [kw.strip() for kw in m.group(1).split(",") if kw.strip()]
            
            if not keywords:
                st.error("No keywords extracted from prompt.")
                return False
            
            base_date = datetime.now()
            related_news = self.search_related_news(keywords, base_date)
            
            if not related_news:
                st.error("No relevant news articles found for extracted keywords.")
                return False
            
            # Create initial event (maintain all original parameters)
            initial_news = related_news[0]
            initial_event = NewsEvent(
                title=initial_news['title'],
                url=initial_news['url'],
                date=initial_news['date'],
                summary=initial_news['summary'],
                keywords=keywords,
                relevance_score=1.0,
                depth_level=0
            )
            
            self.graph.add_node(
                initial_news['url'],
                event=initial_event,
                pos=(0, 0, 0),
                color='red',
                size=30
            )
            
            # Build graph using BFS with memory management
            queue = deque([(initial_event, 0)])
            processed = {initial_news['url']}
            progress_bar = st.progress(0)
            total_operations = self.max_depth * self.max_events_per_level
            current_operation = 0
            
            while queue and current_operation < total_operations:
                current_event, depth = queue.popleft()
                if depth >= self.max_depth:
                    continue
                
                # Periodic memory cleanup
                if current_operation % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Search for related news (maintain original logic)
                related_news = self.search_related_news(current_event.keywords, current_event.date)
                
                # Process related articles (maintain all original parameters)
                added_count = 0
                for news in related_news[:self.max_events_per_level]:
                    if news['url'] in processed or added_count >= 5:
                        continue
                    
                    relevance = self.calculate_relevance_score(news, current_event.keywords)
                    if relevance < 0.3:
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
                    
                    # Calculate position (maintain original positioning logic)
                    angle = (added_count * 360 / 8) * np.pi / 180
                    radius = (depth + 1) * 2
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
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
                    time.sleep(0.1)
            
            progress_bar.empty()
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            # Emergency cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            st.error(f"Error building event graph from prompt: {str(e)}")
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