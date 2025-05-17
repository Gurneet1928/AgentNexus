import lmstudio as lms
from lmstudio.sync_api import DownloadedLlm
import yaml
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
import random

if 'static_flow_state' not in st.session_state:
    st.session_state.static_flow_state = StreamlitFlowState([], [])

def get_thinking_message():
    """Return a random thinking message from the AI's perspective"""
    thinking_messages = [
        "Diving into the digital cosmos for answers... 🧠✨",
        "Consulting my silicon neurons... processing in progress! 🤖",
        "Running through billions of possibilities at lightspeed... ⚡",
        "Hmm, let me connect some neural pathways for this one... 🔄",
        "Extracting knowledge from the digital realm... please stand by! 🌐",
        "Ahh, one more query. Let me think about it... 🤔",
        "Just a moment, I'm sifting through the data... ⏳",
        "Engaging in a bit of digital soul-searching... 💭"
        "Let me put on my thinking cap... or should I say, my thinking circuit? 🎩",
    ]
    return random.choice(thinking_messages)

def listLlms() ->  dict:
    DownloadModels = lms.list_downloaded_models()
    llms = {}
    for model in DownloadModels:
        if isinstance(model, DownloadedLlm):
            llms[model.model_key] = model
    return llms

@st.dialog("Model Reasoning Flow", width='large')
def getFlow(query, output, intermediate_steps):
    st.markdown("### Agent's Reasoning Process:")
    
    # Define consistent node styles with increased padding
    input_style = {
        'color': 'black', 
        'backgroundColor': '#e6f7ff', 
        'border': '2px solid #1890ff',
        'borderRadius': '8px',
        'padding': '20px',  # Increased padding
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'width': '300px',   # Fixed width for consistent sizing
        'minHeight': '120px' # Minimum height
    }
    
    tool_style = {
        'color': 'black', 
        'backgroundColor': '#f6ffed', 
        'border': '2px solid #52c41a',
        'borderRadius': '8px',
        'padding': '20px',  # Increased padding
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'width': '300px',   # Fixed width
        'minHeight': '150px' # Minimum height
    }
    
    output_style = {
        'color': 'black', 
        'backgroundColor': '#fff1f0', 
        'border': '2px solid #ff4d4f',
        'borderRadius': '8px',
        'padding': '20px',  # Increased padding
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
        'width': '300px',   # Fixed width
        'minHeight': '120px' # Minimum height
    }
    
    # Increase spacing between nodes
    horizontal_spacing = 400  # Increased from 250
    vertical_spacing = 250    # Increased from 180
    
    # Start with query node
    nodes = [
        StreamlitFlowNode(
            id='query', 
            pos=(100, vertical_spacing), 
            data={'content': f"**User Query**: {query}"}, 
            node_type='input', 
            source_position='right', 
            draggable=False,
            style=input_style
        )
    ]
    
    edges = []
    
    # Create tool nodes with plain text formatting (no HTML)
    for i, step in enumerate(intermediate_steps):
        action = step[0]  # ToolAgentAction object
        observation = step[1]  # Tool result
        
        node_id = f'tool_{i}'
        tool_name = action.tool.replace('_', ' ').title()
        
        # Format tool input (no HTML tags)
        input_display = ""
        if isinstance(action.tool_input, dict):
            for k, v in action.tool_input.items():
                input_display += f"{k}: {v}\n"
        else:
            input_display = f"{str(action.tool_input)}"
        
        # Create tool node with plain text formatting
        nodes.append(
            StreamlitFlowNode(
                id=node_id,
                pos=(100 + horizontal_spacing, vertical_spacing * (i+1)), 
                data={'content': f"**Tool**: {tool_name}\n\n**Input**: {input_display}\n\n**Result**: {observation}"}, 
                node_type='default', 
                source_position='right', 
                target_position='left', 
                draggable=False,
                style=tool_style
            )
        )
        
        # Connect nodes with non-animated edges
        if i == 0:
            # First tool connects to query
            edges.append(
                StreamlitFlowEdge(
                    id=f'edge_query_{node_id}', 
                    source='query', 
                    target=node_id,
                    animated=False,
                    style={'stroke': '#1890ff', 'strokeWidth': 2},
                    marker_end={'type': 'arrowclosed', 'color': '#1890ff'},
                    label="Process"
                )
            )
        else:
            # Other tools connect to previous tool
            prev_node_id = f'tool_{i-1}'
            edges.append(
                StreamlitFlowEdge(
                    id=f'edge_{prev_node_id}_{node_id}', 
                    source=prev_node_id, 
                    target=node_id, 
                    animated=False,
                    style={'stroke': '#52c41a', 'strokeWidth': 2},
                    marker_end={'type': 'arrowclosed', 'color': '#52c41a'},
                    label="Next step"
                )
            )
    
    # Add final response node
    response_id = 'response'
    nodes.append(
        StreamlitFlowNode(
            id=response_id,
            pos=(100 + horizontal_spacing * 2, vertical_spacing * ((len(intermediate_steps) // 2) + 1)), 
            data={'content': f"**Final Response**: {output}"}, 
            node_type='output', 
            target_position='left', 
            draggable=False,
            style=output_style
        )
    )
    
    # Connect last tool to response or query to response if no tools
    if intermediate_steps:
        last_tool_id = f'tool_{len(intermediate_steps)-1}'
        edges.append(
            StreamlitFlowEdge(
                id=f'edge_{last_tool_id}_{response_id}', 
                source=last_tool_id, 
                target=response_id, 
                animated=False,
                style={'stroke': '#ff4d4f', 'strokeWidth': 2},
                marker_end={'type': 'arrowclosed', 'color': '#ff4d4f'},
                label="Final result"
            )
        )
    else:
        # If no tools were used, connect query directly to response
        edges.append(
            StreamlitFlowEdge(
                id=f'edge_query_{response_id}', 
                source='query', 
                target=response_id, 
                animated=False,
                style={'stroke': '#ff4d4f', 'strokeWidth': 2},
                marker_end={'type': 'arrowclosed', 'color': '#ff4d4f'},
                label="Direct response"
            )
        )
    
    st.session_state.static_flow_state = StreamlitFlowState(nodes, edges)
    
    streamlit_flow(
        'static_flow',
        st.session_state.static_flow_state,
        fit_view=True,
        show_minimap=True,
        show_controls=True,
        pan_on_drag=True,
        allow_zoom=True,
        hide_watermark=True,
        height=700  # Increased height
    )
# Count the characters in the word "HTML" and store them in a file hello.txt 
