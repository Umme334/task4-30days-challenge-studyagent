# Gemini Integration Reference

This document compiles the relevant code snippets and configurations for the Gemini model integration, including OpenRouter setup, direct model wrappers, and schema conversion utilities.

## Configuration (OpenRouter)

Configuration for using Gemini 2.0 Flash via OpenRouter.

```python
def setup_gemini_config():
    """
    Create a custom evaluation configuration using Gemini 2.0 Flash via OpenRouter
    """
    # Configure to use Gemini 2.0 Flash via OpenRouter
    evaluation_config = {
        "model_name": "google/gemini-2.0-flash-001",  # OpenRouter format for Gemini
        "provider": "openai_endpoint",  # Use OpenRouter as endpoint
        "openai_endpoint_url": "https://openrouter.ai/api/v1",
        "temperature": 0,  # Zero temp for consistent evaluation
    }

    print(f"Using Gemini 2.0 Flash for evaluation: {evaluation_config}")
    return evaluation_config

def setup_gemini_config_with_key(api_key):
    """
    Create a configuration for using Gemini via OpenRouter.

    Args:
        api_key: OpenRouter API key

    Returns:
        Dictionary with configuration settings
    """
    return {
        "model_name": "google/gemini-2.0-flash-001",
        "provider": "openai_endpoint",
        "openai_endpoint_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
    }
```

## Gemini Model Wrapper

A custom wrapper class for the Gemini model handling initialization, retries, and parallel execution.

```python
class GeminiModel:
    """Class for the Gemini model."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        finetuned_model: bool = False,
        distribute_requests: bool = False,
        cache_name: str | None = None,
        temperature: float = 0.01,
        **kwargs,
    ):
        self.model_name = model_name
        self.finetuned_model = finetuned_model
        self.arguments = kwargs
        self.distribute_requests = distribute_requests
        self.temperature = temperature
        model_name = self.model_name
        if not self.finetuned_model and self.distribute_requests:
            random_region = random.choice(GEMINI_AVAILABLE_REGIONS)
            model_name = GEMINI_URL.format(
                GCP_PROJECT=GCP_PROJECT,
                region=random_region,
                model_name=self.model_name,
            )
        if cache_name is not None:
            cached_content = caching.CachedContent(cached_content_name=cache_name)
            self.model = GenerativeModel.from_cached_content(
                cached_content=cached_content
            )
        else:
            self.model = GenerativeModel(model_name=model_name)

    @retry(max_attempts=12, base_delay=2, backoff_factor=2)
    def call(self, prompt: str, parser_func=None) -> str:
        """Calls the Gemini model with the given prompt."""
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                **self.arguments,
            ),
            safety_settings=SAFETY_FILTER_CONFIG,
        ).text
        if parser_func:
            return parser_func(response)
        return response

    def call_parallel(
        self,
        prompts: List[str],
        parser_func: Optional[Callable[[str], str]] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ) -> List[Optional[str]]:
        """Calls the Gemini model for multiple prompts in parallel using threads with retry logic."""
        results = [None] * len(prompts)

        def worker(index: int, prompt: str):
            """Thread worker function to call the model and store the result with retries."""
            retries = 0
            while retries <= max_retries:
                try:
                    return self.call(prompt, parser_func)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error for prompt {index}: {str(e)}")
                    retries += 1
                    if retries <= max_retries:
                        print(f"Retrying ({retries}/{max_retries}) for prompt {index}")
                        time.sleep(1)  # Small delay before retrying
                    else:
                        return f"Error after retries: {str(e)}"

        # Create and start one thread for each prompt
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_index = {
                executor.submit(worker, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Unhandled error for prompt {index}: {e}")
                    results[index] = "Unhandled Error"

        # Handle remaining unfinished tasks after the timeout
        for future in future_to_index:
            index = future_to_index[future]
            if not future.done():
                print(f"Timeout occurred for prompt {index}")
                results[index] = "Timeout"

        return results
```

## Core Integration Class

Integration implementation for Gemini models, including async generation and streaming support.

```python
class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
  """

  model: str = 'gemini-1.5-flash'

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models."""
    return [
        r'gemini-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model."""
    self._preprocess_request(llm_request)
    self._maybe_append_user_content(llm_request)
    # ... (Logging and request setup) ...

    if stream:
      responses = await self.api_client.aio.models.generate_content_stream(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      # ... (Stream handling logic) ...
      async for response in responses:
          # ... (Yield LlmResponse) ...
          yield LlmResponse.create(response) # Simplified
    else:
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      yield LlmResponse.create(response)

  # ... (Helper properties and methods: api_client, _api_backend, connect, _preprocess_request) ...
```

## Schema Conversion Utilities

Functions to convert between Gemini Schema objects and JSON Schema dictionaries.

```python
def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary."""
  if not isinstance(gemini_schema, Schema):
    raise TypeError(f"Input must be an instance of Schema, got {type(gemini_schema)}")

  json_schema_dict: Dict[str, Any] = {}

  # Map Type
  gemini_type = getattr(gemini_schema, "type", None)
  if gemini_type and gemini_type != Type.TYPE_UNSPECIFIED:
    json_schema_dict["type"] = gemini_type.lower()
  else:
    json_schema_dict["type"] = "null"

  # ... (Mapping Nullable, direct fields, String/Number/Array/Object validation) ...

  return json_schema_dict

def _to_gemini_schema(openapi_schema: dict[str, Any]) -> Schema:
  """Converts an OpenAPI schema dictionary to a Gemini Schema object."""
  if openapi_schema is None:
    return None

  if not isinstance(openapi_schema, dict):
    raise TypeError("openapi_schema must be a dictionary")

  openapi_schema = _sanitize_schema_formats_for_gemini(openapi_schema)
  return Schema.from_json_schema(
      json_schema=_ExtendedJSONSchema.model_validate(openapi_schema),
      api_option=get_google_llm_variant(),
  )
```

## Tests

Unit tests for schema conversion.

```python
class TestToGeminiSchema:
  # ... (Tests for various schema scenarios: basic types, nested objects, arrays, formats, etc.) ...
  def test_to_gemini_schema_basic_types(self):
    openapi_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert isinstance(gemini_schema, Schema)
    assert gemini_schema.type == Type.OBJECT
    # ... (assertions) ...
```

## Additional Snippets

### Factory & Mock

```python
def gemini_llm():
  return Gemini(model="gemini-1.5-flash")

def mock_gemini_session():
  """Mock Gemini session for testing."""
  return mock.AsyncMock()
```

### Google Search Tool Note

> This tool is automatically invoked by Gemini models to retrieve Google Search results. It configures the model's request to include the necessary Google Search functionality. It handles model version compatibility, ensuring correct tool integration for Gemini 1.x and Gemini 2.
