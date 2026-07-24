## steps
- adk create my_agent
- run options:
  - adk run my_agent: terminal
  - adk web --port 8000: web UI
  - adk api_server: to expose API

## Notes
- agent.py file in the root agent directory contains the main control and is the only required file

## Features
- tools:
  - agent relies heavily on the docstring/comment of the tool function definition to use it correctly 
- auto flow: 
  - automatic delegation to sub_agents 
  - delegating user requests to the appropriate specialized sub-agent
  - using sub_agents parameter in the Agent() constructor
- Session State and ToolContext:
  - helps agent retain information across conversational turns for more contextual chat
  - it is a python dictionary (session.state) tied to a specific user session (identified by APP_NAME, USER_ID, SESSION_ID)
  - it persists information across multiple conversational turns within that session
  - agents can read from and write to it allowing them to remember context and personalize response
  - how agents interact with state?
    - ToolContext (Primary Method): Tools can accept a ToolContext object and it has tool_context.state which is the state
    - output_key="your_key": ADK will automatically save agent's final text response for a turn into session.state["your_key"] 
- before_model_callback, before_tool_callback:
  - guardrails
- agents:
  - description of agents is used by other agents to decide if this agent needs to be called for a specific task
  - to manage conversations and execute the agent we need 2 more components:
    - SessionService: 
      - to manage conversation state and history across different users and sessions
      - InMemorySessionService: stores everything in memory; useful for simple apps and testing
    - Runner:
      - the engine that orchestrates the interaction flow
      - it takes input, routes it to the appropriate agent, manages calls to the llm and tools, handles session updates for SessionService, and yields the state of the conversation
      - it operates asynchronously since call to llm and tool takes time
- LiteLlm wrapper: to use various models without much code change
- 