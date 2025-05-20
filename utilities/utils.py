import yaml
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
import random
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path

if 'static_flow_state' not in st.session_state:
    st.session_state.static_flow_state = StreamlitFlowState([], [])

def read_yaml(path_to_yaml: Path) -> ConfigBox:  # Input Arguments -> Output Argument Type
    """
    Reads yaml file and returns
    Args:
        path_to_yaml: Path input
    Raises:
        ValueError: If file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox Type
    """
    try:
        with open(path_to_yaml, 'r', encoding="utf-8") as file:
            content = ConfigBox(yaml.safe_load(file))
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Empty file: {path_to_yaml}")
    except Exception as e:
        return e

def get_thinking_message():
    """
    Return a random thinking message from the AI's perspective
    Args:
        None
    Raises:
        None
    Returns:
        str: A random thinking message
    """
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

@st.dialog("Model Reasoning Flow", width='large')
def getFlow(query, output, intermediate_steps):
    """
    Visualize the reasoning process of the model using Streamlit Flow and display using st.dialog()
    Args:
        query (str): The user's query
        output (str): The final output from the model
        intermediate_steps (list): A list of intermediate steps taken by the model
    Raises:
        None
    Returns:
        None
    """
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
    
    horizontal_spacing = 400  
    
    # Use a consistent y position for all nodes
    base_y_pos = 250
    
    # Start with query node
    nodes = [
        StreamlitFlowNode(
            id='query', 
            pos=(100, base_y_pos), 
            data={'content': f"**User Query**: {query}"}, 
            node_type='input', 
            source_position='right', 
            draggable=False,
            style=input_style
        )
    ]

    edges = []
    
    for i, step in enumerate(intermediate_steps):
        action = step[0]  # ToolAgentAction object
        observation = step[1]  # Tool result
        
        node_id = f'tool_{i}'
        tool_name = action.tool.replace('_', ' ').title()
        
        input_display = ""
        if isinstance(action.tool_input, dict):
            for k, v in action.tool_input.items():
                input_display += f"{k}: {v}\n"
        else:
            input_display = f"{str(action.tool_input)}"

        # Simply place each node to the right of the previous one
        x_pos = 100 + horizontal_spacing * (i + 1)
        
        # All nodes have the same y position
        y_pos = base_y_pos
        
        nodes.append(
            StreamlitFlowNode(
                id=node_id,
                pos=(x_pos, y_pos), 
                data={'content': f"**Tool**: {tool_name}\n\n**Input**: {input_display}\n\n**Result**: {observation}"}, 
                node_type='default', 
                source_position='right', 
                target_position='left', 
                draggable=False,
                style=tool_style
            )
        )    
        
        # Connect edges same as before
        if i == 0:
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
    
    # Position response node at the end of the line
    if intermediate_steps:
        response_x = 100 + horizontal_spacing * (len(intermediate_steps) + 1)
    else:
        response_x = 100 + horizontal_spacing
    
    response_y = base_y_pos

    response_id = 'response'
    nodes.append(
        StreamlitFlowNode(
            id=response_id,
            pos=(response_x, response_y), 
            data={'content': f"**Final Response**: {output}"}, 
            node_type='output', 
            target_position='left', 
            draggable=False,
            style=output_style
        )
    )
    
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
        height=700
    )