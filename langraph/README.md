## Introduction
- used to make building llm apps like agents/multi-agent-systems easier
- agents are built by allowing an llm to determine to control flow of an app to automate tasks
- challenge is to make the agent run reliably and with precision. this is because llms are probabilistic in nature. we might need determinism to always call a particular tool
- Langraph studio: desktop IDE in developing and debugging langraph apps
- Router: an llm can act as a router in choosing one of many steps based on it's state (created in response to the user's input)
  - as we make the apps more autonomous the reliability decreases.  Langraph helps in increasing the reliability
  - pillars of langraph that helps us achieve this reliability: persistence/in-memory, humann in the loop, controllability, streaming
  - for example, a llm model can decide between returning a tool_call or a natural language answer as the response.  Thus it acts as a router in this case 
    - the tool call here is performed by a node that will call our tool, or we can simply use the built-in ToolNode
      - Ex: builder.add_node("tools", ToolNode(["tool_function_name"]))
    - to call either the toolNode or simply respond with natural language we do this:
      - builder.add_conditional_edge("tool_calling_llm", tools_condition) # tools_condition is built-in
- Langraph works nicely with langchain (which has many llm integrations, vector store integrations etc.).

## Graph
- Nodes are the steps in our app like tool-call or retrieval step etc., and edges are the connectivity among these nodes
- Node structure START Node_0 -> Node_1 -> END
                               -> Node_2 ->
  - START to Node_1 OR Node_2 is called conditional edge
- To define a graph in Langraph, we need to first define a state
  - The state is the object that we pass between the nodes and edges of the graph.
  - class State(TypeDict): //State is of type dictionary
    - state_attribute: str
- nodes are defined as simple python functions that takes in state defined above: def node_1(State):
  - we might update the state_attribute in this node, and return it for this new state to be available to the next nodes
- edges: to connect the nodes
  - normal edge
  - conditional edge: just a normal python function
    - def conditional_edge(state):
      - if(x): return node_1
      - else: return node_2
- graph:
  - builder = langraph.graph.StateGraph(State)
  - builder.add_node("node_1", node_1) # similarly define other nodes
  - builder.add_edge(START, node_1)
  - builder.add_conditional_edge(node_1, conditional_edge) # this conditional_edge is the function defined above
  - graph = builder.compile()
  - graph.invoke({"state_attribute": "initial"}) # this state_attribute is a key in the State TypeDict defined above
    - this runs our graph
- tools:
  - define as a normal python function
  - llm = langchain.chatopenai('chatgpt')
  - llm_with_tools = llm.bind_tools(tool_function_name) #ex tool_function_name = multiply
    - natural language input is converted to tool call by identifying and setting the right parameters to this tool function
- reducers:
  - reducers tell us how to update the state as it passes through different nodes, for example append to the list of messages each time a new message appears, whereas normally we just overwrite the whole state
  - we can use class State(MessageState) which will have this functionality by default i.e. all the to-and-fro HumanMessage and AIMessage will be appended to the list 'messages'
- Chain:
  - # Node
  - def llm_calling_node(state: MessagesState):
    - return {'messages': [llm_with_tools.invoke(state['messages''])]} # llm_with_tools is defined above 
  - now if we defina a graph using this node, and call the graph with input: "what is 2 multiplied by 3?", it will invoke the tool call that multiplies the numbers, and respond with this tool_call output

## Other concepts
- threads: to capture history of any run of the graph
- response_metadata gives us extra info like token usage, model used, etc.
- Agent Architectures:
  - React: We take the result from the tool call and pass it back to the model
    - start -> llm_router_tool_node <-> tool_call
                  |
                  v
               final natural language answer to user 
    - Ex. if tool call output is 12 then final natural language output will be: the multiplication of 3 and 4 is 12
    - to-and-fro can happen between llm_router_tool_node and tool_call multiple times (in a loop) until the llm_router_tool_node deems the output from the tool_call satisfactory 
      - we can set a max_limit for the number of times the tool can be called
    - steps:
      - act: model call tools
      - observe: reason based on tool output what to do next
      - reason: either to stop or call more tools until the goal is reached
    - we need to add an edge from the tool_node back to the assistant in addition to add_conditional_edge with tools_condition above 
    - Ex: multiply 2 and 3 and then add 5 to the output
      - here 2 tools might be called - multiply and add
- Memory:
  - we need it to retain working memory when we do new invocations to the graph multiple times. This is because the state is transient between 2 graph invocations 
  - For example: "add 2 and 3" followed by "add 5 to it".  In this case, the agent will not remember the first conversation and will not know what to add 5 to
  - langchain uses checkpointer to save the graph state after each step.  One of the easiest checkpointers is MemorySaver, which is just an in-memory key-value store
    - graph = builder.compile(checkpointer=MemorySaver())
    - The checkpointer writes the current state to memory after every step in our graph 
    - The checkpoints can be associated together in a thread.  A thread is a collection of checkpoints
      - config = {"configurable": {"thread_id": "1"}}
      - graph.invoke({"messages": messages}, config)
    - Now when I say graph.invoke for "add 5 to it", we need to pass this same config with the same thread it for it to be able to access the previous invocation of "add 2 and 3".  So, it will add 5 to 5 (answer of "add 2 and 3")
    - In studio we don't need to supply checkpointer because studio is backed by the langraph API, which packages the code for us and it has it's own persistence layer (postgres)

## Deployment
- Langraph: python and js library to build and run agents
- langraph API: 
  - bundles the graph code
  - offers persistence to maintain state across interactions
  - provides a task queue to to manage async ops
- langraph Cloud:
  - hosted service for the langraph API
  - allows deployment of graphs from github repo
  - offers monitoring, tracing and API documentation
- langraph SDK:
  - python library to interact with langraph graphs by providing HumanMessages