import streamlit as st
import pandas as pd
from main import NewsEventMapper
from datetime import datetime
import json
import numpy as np

import llm
tokenizer, model = llm.model_init()

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
        user_prompt = st.text_area(
            "Enter a news-related prompt or description:",
            placeholder="E.g. 'Recent developments in AI regulation in Europe'",
            help="Describe a news topic, event, or area of interest to analyze its background and connections"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze", type="primary")

    # Sample prompts for testing
    with st.expander("📝 Try Sample Prompts"):
        sample_prompts = [
            "US presidential election 2024 controversies",
            "Major breakthroughs in cancer research 2025",
            "Global impact of electric vehicle adoption"
        ]
        for i, prompt in enumerate(sample_prompts):
            if st.button(f"Sample {i+1}: {prompt}", key=f"sample_{i}"):
                st.session_state.sample_prompt = prompt
                user_prompt = prompt

    # Process analysis
    if analyze_button and user_prompt:
        with st.spinner("🔍 Analyzing prompt and building connection map..."):
            st.info("📡 Extracting keywords and searching news articles...")
            success = st.session_state.mapper.build_event_graph_from_prompt(user_prompt, model, tokenizer)
            if success:
                st.session_state.graph_built = True
                st.success("✅ Analysis complete! Explore the mind map below.")
            else:
                st.error("❌ Failed to analyze the prompt. Please try a different topic or description.")
    
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