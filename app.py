import streamlit as st
import pandas as pd
from main import NewsEventMapper
from datetime import datetime
import json
import numpy as np
import torch
import gc
from PromptEngine import ScriptEngine
import heapq

import llm
if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
    with st.spinner("Loading AI models..."):
        tokenizer, model = llm.model_init()
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
else:
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

def main():
    TOP_K = 10

    st.set_page_config(
        page_title="News Event Mind Map Generator",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("MERT")
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
    
    # Memory Management Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Memory Management")
    if st.sidebar.button("🧹 Clear GPU Memory"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        st.sidebar.success("Memory cleared!")
    
    # GPU Memory Display
    if torch.cuda.is_available():
        try:
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            st.sidebar.metric("GPU Memory", f"{memory_used:.1f}GB / {memory_total:.1f}GB")
        except:
            pass  # Skip if CUDA info unavailable
    
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

    # Process analysis with enhanced error handling
    if analyze_button and user_prompt:
        try:
            with st.spinner("🔍 Analyzing prompt and building connection map..."):
                st.info("📡 Extracting keywords and searching news articles...")
                success = st.session_state.mapper.build_event_graph_from_prompt(
                    user_prompt, 
                    st.session_state.model, 
                    st.session_state.tokenizer
                )
                
                # Memory cleanup after analysis
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if success:
                    st.session_state.graph_built = True
                    st.success("✅ Analysis complete! Explore the mind map below.")
                else:
                    st.error("❌ Failed to analyze the prompt. Please try a different topic or description.")
                    
        except torch.cuda.OutOfMemoryError:
            st.error("🔥 CUDA out of memory. Try reducing analysis depth or restarting the app.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
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
                if st.button("📊 Create Script"):
                    with st.spinner("🎬 Generating script..."):
                        try:
                            # Pre-clear GPU memory before intensive operation
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            gc.collect()
                            
                            # graph_data = {
                            #     'nodes': [
                            #         {
                            #             'id': node,
                            #             'title': st.session_state.mapper.graph.nodes[node]['event'].title,
                            #             'date': st.session_state.mapper.graph.nodes[node]['event'].date.isoformat(),
                            #             'relevance': st.session_state.mapper.graph.nodes[node]['event'].relevance_score,
                            #             'depth': st.session_state.mapper.graph.nodes[node]['event'].depth_level,
                            #             'url': st.session_state.mapper.graph.nodes[node]['event'].url
                            #         }
                            #         for node in st.session_state.mapper.graph.nodes()
                            #     ],
                            #     'edges': [
                            #         {
                            #             'source': edge[0],
                            #             'target': edge[1],
                            #             'weight': st.session_state.mapper.graph.edges[edge].get('weight', 1.0),
                            #             'relationship': st.session_state.mapper.graph.edges[edge].get('relationship', 'related')
                            #         }
                            #         for edge in st.session_state.mapper.graph.edges()
                            #     ]
                            # }
                            all_nodes = [
                                {
                                    'id': node,
                                    'title': st.session_state.mapper.graph.nodes[node]['event'].title,
                                    'date': st.session_state.mapper.graph.nodes[node]['event'].date.isoformat(),
                                    'relevance': st.session_state.mapper.graph.nodes[node]['event'].relevance_score,
                                    'depth': st.session_state.mapper.graph.nodes[node]['event'].depth_level,
                                    'url': st.session_state.mapper.graph.nodes[node]['event'].url
                                }
                                for node in st.session_state.mapper.graph.nodes()
                            ]

                            # Step 2: Select top-K relevant nodes
                            top_nodes = heapq.nlargest(TOP_K, all_nodes, key=lambda x: x['relevance'])

                            # Step 3: Get set of selected node IDs
                            selected_node_ids = set(node['id'] for node in top_nodes)

                            # Step 4: Filter edges that connect selected nodes
                            filtered_edges = [
                                {
                                    'source': edge[0],
                                    'target': edge[1],
                                    'weight': st.session_state.mapper.graph.edges[edge].get('weight', 1.0),
                                    'relationship': st.session_state.mapper.graph.edges[edge].get('relationship', 'related')
                                }
                                for edge in st.session_state.mapper.graph.edges()
                                if edge[0] in selected_node_ids and edge[1] in selected_node_ids
                            ]

                            # Step 5: Create graph data
                            graph_data = {
                                'nodes': top_nodes,
                                'edges': filtered_edges
                            }

                            final_prompt = "The original prompt of the user was: " + user_prompt + "\n\n" + \
                                "Generate a script according to the given json file. "+ json.dumps(graph_data)
                            
                            # Temporary move model to CPU during script generation if needed
                            model_was_on_cuda = next(st.session_state.model.parameters()).is_cuda
                            if model_was_on_cuda and torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.8:
                                st.session_state.model.cpu()
                                torch.cuda.empty_cache()
                            
                            script_engine = ScriptEngine(st.session_state.model, st.session_state.tokenizer, False)
                            script = script_engine.generate_script(prompt=final_prompt)
                            
                            # Move model back to GPU if it was there originally
                            if model_was_on_cuda and not next(st.session_state.model.parameters()).is_cuda:
                                st.session_state.model.cuda()
                            
                            st.session_state.generated_script = script
                            st.success("✅ Script generated successfully!")
                            
                        except torch.cuda.OutOfMemoryError:
                            st.error("🔥 CUDA out of memory. Attempting CPU fallback...")
                            try:
                                # Force CPU execution
                                st.session_state.model.cpu()
                                torch.cuda.empty_cache()
                                script_engine = ScriptEngine(st.session_state.model, st.session_state.tokenizer, False)
                                script = script_engine.generate_script(prompt=json.dumps(graph_data))
                                st.session_state.generated_script = script
                                st.success("✅ Script generated on CPU!")
                                # Move back to GPU
                                if torch.cuda.is_available():
                                    st.session_state.model.cuda()
                            except Exception as cpu_e:
                                st.error(f"❌ CPU fallback failed: {str(cpu_e)}")
                                
                        except Exception as e:
                            st.error(f"❌ Script generation failed: {str(e)}")
                        
                        finally:
                            # Always cleanup
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                
                # Show download button only if script exists
                if 'generated_script' in st.session_state:
                    st.download_button(
                        label="📥 Download Script",
                        data=st.session_state.generated_script,
                        file_name=f"news_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
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
    

if __name__ == "__main__":
    main()