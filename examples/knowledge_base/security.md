# Security

## Concept

Security protects the agent from unsafe actions, prompt injection, unauthorized file access, and uncontrolled tool execution.

## Components

A secure agent platform should include:

- Permission System: controls which tools can run and what resources they can access.
- Workspace Boundary: prevents tools from reading or writing files outside the allowed workspace.
- Tool Input Validation: validates all tool arguments before execution.
- Trace Events: record sensitive operations for debugging and audit.
- Policy Checks: reject unsafe or unsupported actions.

## Permission System

The permission system decides whether a tool call is allowed before the tool is executed. It can check tool name, input arguments, workspace path, user approval, and runtime policy.

## Trace Events

Trace events should record important runtime actions, including model requests, model responses, tool calls, tool results, errors, and stop reasons.

## Design Notes

Security should be implemented in the runtime and tool execution layer, not only in prompts.