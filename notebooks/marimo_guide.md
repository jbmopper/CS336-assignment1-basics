# Marimo Code Guide (source-based)

This guide summarizes marimo's core API and behavior by reading the library
source. It is meant as a quick reference for writing marimo notebooks without
needing to guess how reactivity works.

**Important**: Marimo notebooks are stored as pure Python files (`.py`), not
JSON like Jupyter. Each cell is a function decorated with `@app.cell`.

## Mental model

- A marimo notebook is an `App` that builds a dataflow graph.
- Each `@app.cell` function is a node. Its **parameters** are its dependencies
  (references), and its **return values** are its definitions.
- When a dependency changes, downstream cells re-run automatically.
- The last expression in a cell is its visual output; you can also use
  `mo.output.append/replace/clear` for multi-output cells.
- **Cells are functions, not scripts** - they have exactly ONE return statement.

## Minimal template

```python
import marimo as mo

app = mo.App()

with app.setup:
    import pandas as pd

@app.cell
def _(mo, pd):
    df = pd.DataFrame({"x": [1, 2, 3]})
    mo.md(f"Rows: {len(df)}")
    return (df,)

@app.cell
def _(df):
    df.describe()

if __name__ == "__main__":
    app.run()
```

## App configuration

`mo.App(...)` accepts program-level config (via `_AppConfig`), for example:

```python
app = mo.App(
    width="full",                 # "compact", "medium", "full", "columns"
    app_title="My App",
    layout_file="layout.json",    # relative to notebook file
    css_file="styles.css",        # relative to notebook file
    html_head_file="head.html",   # relative to notebook file
    auto_download=["html"],       # auto-export on run
    sql_output="auto",            # SQL output type
)
```

## Critical syntax rules (common mistakes)

### Rule 1: ONE return statement per cell

Each cell is a Python function. Functions can only have ONE return statement
that executes. **Do NOT write multiple return statements.**

```python
# WRONG - multiple returns
@app.cell
def _(mo):
    x = 1
    return (x,)
    y = 2        # Never executes!
    return (y,)  # Never executes!

# WRONG - conditional returns that confuse the graph
@app.cell
def _(condition):
    if condition:
        return (a,)
    else:
        return (b,)  # Graph can't statically determine outputs!

# CORRECT - single return at end
@app.cell
def _(mo):
    x = 1
    y = 2
    return x, y
```

### Rule 2: Return tuples, not bare values

Returns must be tuples. A single value needs a trailing comma.

```python
# WRONG - bare value
@app.cell
def _():
    x = 42
    return x  # Not a tuple!

# WRONG - forgetting trailing comma
@app.cell
def _():
    x = 42
    return (x)  # This is just x in parentheses, not a tuple!

# CORRECT - trailing comma for single values
@app.cell
def _():
    x = 42
    return (x,)  # Tuple with one element

# CORRECT - multiple values (comma separates them)
@app.cell
def _():
    x = 42
    y = "hello"
    return x, y  # Parentheses optional for multiple values
```

### Rule 3: Function parameters declare dependencies

The cell's function parameters must **exactly match** variable names defined
by other cells. This is how marimo builds the dependency graph.

```python
# Cell 1 defines 'data' and 'config'
@app.cell
def _():
    data = [1, 2, 3]
    config = {"lr": 0.01}
    return data, config

# Cell 2 uses them - parameter names must match!
@app.cell
def _(data, config):  # These names must match exactly
    result = sum(data) * config["lr"]
    return (result,)

# WRONG - parameter name doesn't match any definition
@app.cell
def _(my_data):  # Error! No cell defines 'my_data'
    return (my_data,)
```

### Rule 4: No variable shadowing across cells

Each variable name can only be defined by ONE cell. You cannot redefine
a variable in a different cell.

```python
# Cell 1
@app.cell
def _():
    x = 1
    return (x,)

# WRONG - Cell 2 tries to redefine x
@app.cell
def _():
    x = 2  # Error! 'x' is already defined by another cell
    return (x,)

# CORRECT - use a different name
@app.cell
def _(x):
    x_doubled = x * 2
    return (x_doubled,)
```

### Rule 5: Cells without returns are valid (display-only)

If a cell just displays output and defines nothing, omit the return.

```python
# Display-only cell - no return needed
@app.cell
def _(data, mo):
    mo.md(f"Data has {len(data)} items")
    # No return - this cell defines nothing

# The last expression is automatically displayed
@app.cell
def _(df):
    df.head()  # This DataFrame is displayed automatically
```

### Rule 6: The underscore `_` convention

- `_` as function name means "anonymous cell" (no special meaning to marimo)
- Parameters starting with `_` are still dependencies
- `mo` is conventionally imported in setup and used as a parameter

```python
@app.cell
def _(mo):  # 'mo' comes from setup, '_' is just a placeholder name
    mo.md("Hello")
```

## Setup cell

Use `with app.setup:` to define imports and constants once. This cell is
special: it cannot redefine `app` and cannot reference other variables besides
builtins. You can hide setup code with `with app.setup(hide_code=True):`.

```python
with app.setup:
    import numpy as np
    PI = 3.14159
```

## Cells, definitions, and reuse

- `@app.cell` registers a cell.
- `@app.function` and `@app.class_definition` register a function/class as a
  top-level cell (useful for reusable helpers).
- Cells can be named (function name) and run from outside the notebook:
  `output, defs = named_cell.run()` (await it if async).
- `app.run(defs={...})` overrides entire cell definitions. If you override a
  variable, the **cell that defines it does not run**, and you must provide
  **all** definitions that cell would have produced.

### Async cells

If a cell is `async`, or depends on an async ancestor, `.run()` returns an
awaitable:

```python
from collections.abc import Awaitable

ret = my_cell.run()
if isinstance(ret, Awaitable):
    output, defs = await ret
else:
    output, defs = ret
```

```python
@app.cell
def config():
    batch = 32
    lr = 0.01
    return batch, lr

@app.cell
def train(batch, lr):
    return (batch * lr,)

# From another file:
# outputs, defs = app.run(defs={"batch": 64, "lr": 0.001})
```

## UI elements and reactivity

`mo.ui.*` constructors return `UIElement` objects with a `.value`.

Rules to remember:

- **Never** access `element.value` in the **same cell** that created the
  element. Put the read in another cell.
- You cannot assign to `element.value`. Use `mo.state()` if you need to set
  values imperatively.
- `on_change` must be provided at construction time.

Example:

```python
@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=10, step=1, label="k")
    return (slider,)

@app.cell
def _(slider):
    k = slider.value
    return k
```

## Forms and batching

Any `UIElement` can be turned into a form to gate submission:

```python
prompt = mo.ui.text_area().form()
```

You can also build composite UI from markdown + inputs:

```python
form = (
    mo.md("Name: {name}\nAge: {age}")
      .batch(name=mo.ui.text(), age=mo.ui.number())
      .form()
)
```

## Output helpers

- `mo.md(text)` creates markdown output (supports f-string interpolation).
- `mo.Html("<h1>..</h1>")` renders raw HTML.
- `mo.as_html(obj)` can embed plots/objects into markdown.
- `mo.output.append(value)` appends multiple outputs in one cell.
- `mo.output.replace(value)` replaces the output.
- `mo.output.clear()` clears the output.

```python
@app.cell
def _(mo):
    mo.output.append(mo.md("# Header"))
    mo.output.append(mo.md("More content"))
```

## Output formatting & custom display

Marimo uses a formatting protocol to render objects. Precedence:

1. `_display_()` method (returns any displayable object)
2. Registered formatters (`@mo.formatter` / `@mo.opinionated_formatter`)
3. `_mime_()` method (returns `(mimetype, data)`)
4. `repr` fallback

Example implementing `_mime_`:

```python
class MyThing:
    def _mime_(self):
        return ("text/html", "<b>Hello</b>")
```

## Stdout/stderr capture & redirect

```python
with mo.capture_stdout() as buffer:
    print("Hello")
text = buffer.getvalue()

with mo.redirect_stdout():
    print("This shows up in the cell output")
```

## State (advanced)

`mo.state(initial)` returns `(get_state, set_state)`. Updating state re-runs
cells that reference the getter. Prefer standard dataflow unless you need
shared mutable state.

```python
get_count, set_count = mo.state(0)

@app.cell
def _(get_count):
    return get_count()

@app.cell
def _(set_count):
    # Update via setter only; do not mutate directly
    set_count(lambda v: v + 1)
```

Warnings from source:

- `mo.state` can introduce cycles; use sparingly.
- Do not store `mo.ui` elements inside state.

## Caching (function + block)

Marimo provides cache helpers that are notebook-aware.

### `mo.cache` and `mo.lru_cache` (function decorators)

Key behavior:

- Cache keys include function args **and** closed-over values.
- Cache invalidates when notebook code changes (not just runtime values).
- Works with non-hashable but pickleable arguments.

```python
@mo.cache
def expensive(x):
    return x ** 2

@mo.lru_cache(maxsize=128)
def bounded(x):
    return x ** 2
```

### `mo.cache` and `mo.persistent_cache` (context managers)

Block-level caching must be used at **cell level** (not inside functions).
On cache hit, the block is skipped, so side-effects in the block do **not**
run.

```python
with mo.cache("fast") as cache:
    data = load_data()

with mo.persistent_cache("disk_cache"):
    data = load_data()  # restored from disk on cache hit
```

Notes:

- `mo.persistent_cache` defaults to `__marimo__/cache` relative to the notebook.
- `pin_modules=True` invalidates cache if module versions change.
- `mo.persistent_cache` uses `sys.settrace` and may conflict with debuggers.

## SQL helper

`mo.sql(query, output=True, engine=None)` executes SQL. With no engine, it uses
DuckDB and can reference dataframes in globals.

```python
@app.cell
def _(mo, df):
    mo.sql("SELECT * FROM df WHERE x > 1")
```

Tip: set `MARIMO_SQL_DEFAULT_LIMIT` to auto-limit queries that lack `LIMIT`.

## Watch files and directories

Marimo provides reactive file/directory wrappers:

```python
file_state = mo.watch.file("data.csv")
text = file_state.read_text()

dir_state = mo.watch.directory("data/")
files = list(dir_state.glob("*.csv"))
```

Notes:

- `mo.watch.file()` reacts to file content changes.
- `mo.watch.directory()` reacts to directory structure changes (not contents).
- Both return Path-like wrappers with some methods disallowed.

## Threads & background work

Use `mo.Thread` when you need a background thread that can still send output
to the frontend. Check `should_exit` to stop the thread when the spawning cell
is invalidated.

```python
def target():
    import time
    thread = mo.current_thread()
    while not thread.should_exit:
        time.sleep(1)
        print("tick")

mo.Thread(target=target).start()
```

## App metadata and runtime helpers

- `mo.app_meta().theme` returns `"light"` or `"dark"`.
- `mo.app_meta().mode` returns `"edit"`, `"run"`, `"script"`, `"test"`, or `None`.
- `mo.app_meta().request` returns the current HTTP request (if running as app).
- `mo.query_params()` returns a dict-like object for URL query params.
- `mo.cli_args()` returns a dict-like view of CLI args when running as a script.
- `mo.notebook_dir()` / `mo.notebook_location()` expose the current notebook.
- `mo.running_in_notebook()` tells you if a runtime context is present.
- `mo.refs()` / `mo.defs()` return refs/defs for the currently executing cell.

## Embedding notebooks

You can embed one app inside another:

```python
from my_notebook import app

# In another cell (not the import cell):
result = await app.embed()
result.output  # Html
result.defs    # dict of definitions
```

Use `app.clone()` when you want multiple independent embedded copies.

Notes:

- `app.embed()` cannot be called in the same cell that imports the app.
- `app.embed(defs=...)` does not allow substituting UI elements.

## ASGI apps & islands (advanced)

### ASGI

Use `mo.create_asgi_app()` to serve one or more marimo apps with a server like
`uvicorn`. You can mount multiple apps or even a dynamic directory.

### Islands

`MarimoIslandGenerator` builds reactive HTML snippets for embedding in other
sites. You add code blocks, build, then render `<marimo-island>` HTML and a
script/CSS header via `render_head()`.

## Common antipatterns (what NOT to do)

### Antipattern 1: Treating cells like script blocks

```python
# WRONG - thinking of cells as sequential script chunks
@app.cell
def _():
    # Do step 1
    data = load_data()
    # Do step 2
    processed = process(data)
    # Do step 3
    result = analyze(processed)
    # Do step 4
    save(result)
    return data, processed, result  # Returning everything

# CORRECT - split into separate cells for reactivity
@app.cell
def _():
    data = load_data()
    return (data,)

@app.cell
def _(data):
    processed = process(data)
    return (processed,)

@app.cell
def _(processed):
    result = analyze(processed)
    return (result,)

@app.cell
def _(result):
    save(result)
```

### Antipattern 2: Using global variables instead of parameters

```python
# WRONG - relying on globals
some_global = 42

@app.cell
def _():
    return (some_global * 2,)  # Works but breaks reactivity!

# CORRECT - pass through setup or cell definitions
with app.setup:
    SOME_CONSTANT = 42

@app.cell
def _(SOME_CONSTANT):  # Explicit dependency
    return (SOME_CONSTANT * 2,)
```

### Antipattern 3: Side effects in cells that return values

```python
# WRONG - mixing side effects with return values
@app.cell
def _(data):
    print("Processing...")  # Side effect
    with open("output.txt", "w") as f:  # Side effect
        f.write(str(data))
    processed = transform(data)
    return (processed,)  # This cell re-runs on data change, causing repeated writes!

# CORRECT - separate computation from side effects
@app.cell
def _(data):
    processed = transform(data)
    return (processed,)

@app.cell
def _(processed, mo):
    # Use a button to trigger side effects explicitly
    save_btn = mo.ui.button(label="Save")
    return (save_btn,)

@app.cell
def _(save_btn, processed):
    if save_btn.value:  # Only runs when button clicked
        with open("output.txt", "w") as f:
            f.write(str(processed))
```

### Antipattern 4: Accessing UI value in the same cell

```python
# WRONG - reading .value in same cell as creation
@app.cell
def _(mo):
    slider = mo.ui.slider(0, 100)
    current = slider.value  # Always gets initial value, not reactive!
    return slider, current

# CORRECT - separate creation from reading
@app.cell
def _(mo):
    slider = mo.ui.slider(0, 100)
    return (slider,)

@app.cell
def _(slider):
    current = slider.value  # Reactive - updates when slider changes
    return (current,)
```

## Debugging tips

### Check the dependency graph

In the marimo UI, you can view the dependency graph to see how cells are
connected. If a cell isn't re-running when you expect, check that:

1. The parameter names exactly match the returned variable names
2. You're returning a tuple (with trailing comma for single values)
3. The variable isn't being shadowed

### Use `mo.stop()` for conditional execution

```python
@app.cell
def _(data, mo):
    mo.stop(data is None, mo.md("Waiting for data..."))
    # Code below only runs if data is not None
    result = expensive_computation(data)
    return (result,)
```

### Print statements go to terminal, not notebook

```python
@app.cell
def _(data):
    print(f"Debug: {data}")  # Goes to terminal, not visible in notebook
    mo.output.append(f"Debug: {data}")  # Visible in notebook
```

## Quick reference: Return syntax

| Scenario | Syntax |
|----------|--------|
| No definitions | No return statement |
| One definition | `return (x,)` |
| Two definitions | `return x, y` or `return (x, y)` |
| Many definitions | `return a, b, c, d` |

## Quick reference: Cell types

| Purpose | Pattern |
|---------|---------|
| Define variables | `def _(): ... return (x,)` |
| Transform data | `def _(input): ... return (output,)` |
| Display only | `def _(data, mo): mo.md(...)` (no return) |
| UI element | `def _(mo): return (mo.ui.slider(...),)` |
| Read UI value | `def _(slider): val = slider.value; return (val,)` |
