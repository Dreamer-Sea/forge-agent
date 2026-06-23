# Agent Runtime

## Concept

Agent Runtime is the execution core of forge-agent. It receives a user task, asks a model provider for the next action, executes tool calls, records trace events, and stops when the task is completed or a safety limit is reached.

## Components

The core modules of Agent Runtime include:

- Agent Loop: coordinates model calls and tool execution.
- Model Provider: abstracts model-specific APIs.
- Tool Registry: stores available tools and exposes their schemas.
- Tool Executor: validates and executes tool calls.
- Trace Recorder: records model calls, tool calls, tool results, and stop reasons.
- Settings: controls workspace, max steps, provider configuration, and runtime limits.

## ToolRegistry

ToolRegistry is responsible for registering tools, exposing tool schemas to the model provider, and finding the correct tool implementation when a tool call is requested.

## Design Notes

Agent Runtime should not hard-code business workflows. Capabilities such as file reading, calculation, and knowledge base search should be exposed as tools.