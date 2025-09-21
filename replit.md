# Agent Ops Hub

## Overview

Agent Ops Hub is a minimal AI-powered information retrieval and question-answering system designed for document ingestion, semantic search, and intelligent responses with citation support. The application serves as a foundation for building AI agents that can process web content, store it in a vector database, and provide contextual answers to user queries. It includes a basic skill routing system for handling specific task types like calendar scheduling.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: FastAPI-based REST API with async/await patterns for handling concurrent requests
- **Application Structure**: Modular design with clear separation between API layer, core business logic, and data storage
- **Graph-Based Processing**: Uses LangGraph for building decision trees and routing between different processing paths (Q&A vs skills)
- **State Management**: Pydantic models for type-safe state handling throughout the processing pipeline

### Vector Storage and Retrieval
- **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient similarity search and clustering
- **Embedding Strategy**: Text chunking with overlap (1200 chars, 200 overlap) to preserve context across chunk boundaries
- **Document Storage**: Hybrid approach with FAISS index for vectors and JSONL file for document metadata
- **Search Method**: Cosine similarity with L2 normalization for optimal retrieval performance

### AI Integration
- **LLM Provider**: OpenAI API integration with configurable model selection (defaults to gpt-4o-mini)
- **Processing Pipeline**: Multi-stage workflow including content fetching, chunking, embedding, retrieval, and answer generation
- **Content Processing**: Robust URL fetching with support for HTML, JSON, and plaintext sources
- **Answer Generation**: Context-aware responses with citation tracking for source attribution

### API Design
- **Endpoints**: Two primary endpoints - `/ingest` for adding content and `/ask` for querying
- **Input Flexibility**: Support for both URL-based ingestion and file uploads
- **Response Format**: Structured responses including answers, confidence scores, and source citations
- **Error Handling**: Comprehensive error handling with proper HTTP status codes and logging

### Routing and Skills
- **Intent Classification**: Simple keyword-based routing system to determine processing path
- **Skill System**: Extensible framework for adding specialized capabilities (calendar management as example)
- **Fallback Mechanism**: Default Q&A processing for queries that don't match specific skill patterns

### Configuration Management
- **Environment Variables**: Centralized configuration through .env files for API keys and settings
- **Data Directory**: Configurable storage location with automatic directory creation
- **Model Selection**: Runtime model configuration for different deployment environments

## External Dependencies

### AI and ML Services
- **OpenAI API**: Required for chat completion and text generation capabilities
- **LangChain**: Framework for building language model applications and chains
- **LangGraph**: State-based graph processing for complex AI workflows

### Vector Processing
- **FAISS**: CPU-optimized vector similarity search library
- **NumPy**: Numerical computing for vector operations and array manipulation

### Web and Networking
- **HTTPX**: Async HTTP client for fetching content from URLs with redirect support
- **FastAPI**: Modern web framework for building APIs with automatic documentation
- **Uvicorn**: ASGI server for running the FastAPI application

### Data Processing
- **Pydantic**: Data validation and serialization with type hints
- **Python-multipart**: File upload handling for document ingestion
- **Tiktoken**: OpenAI's tokenizer for text processing and chunking

### Development Tools
- **Rich**: Enhanced terminal output for CLI tools
- **Pytest**: Testing framework for unit and integration tests
- **Python-dotenv**: Environment variable management from .env files