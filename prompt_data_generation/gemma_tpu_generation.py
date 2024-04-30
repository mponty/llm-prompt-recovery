# RUN this inside kaggle kernel!!!
# https://www.kaggle.com/code/dmitriyab/gemma-7b-tpu-generation-2/

print("\n... IMPORTS ...\n")
import os

print("\n... ENVIRONMENT SETUP (LOGGING/KERAS-BACKEND) ...\n")
# Set backend
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation overhead
# Disable logging/warning that may bloat the output

keras_backend = "jax"
allocation_fraction = 0.9
tf_min_log_level = '3'
os.environ["KERAS_BACKEND"] = keras_backend
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(allocation_fraction)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_min_log_level)

# Finish imports
import jax
import keras
import keras_nlp

# Other imports
import time
import copy

print(jax.devices())
print(keras.backend.backend())


# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:25:36.348555Z","iopub.execute_input":"2024-04-10T15:25:36.349162Z","iopub.status.idle":"2024-04-10T15:28:18.396725Z","shell.execute_reply.started":"2024-04-10T15:25:36.349125Z","shell.execute_reply":"2024-04-10T15:28:18.395665Z"}}
# Note we use jax.devices() instead of keras.distribution.list_devices()
def initialize_device_mesh(
        shape: tuple[int, int] = (1, 8),
        batch_axis_name: str = "batch",
        model_axis_name: str = "model"
) -> keras.distribution.DeviceMesh:
    """Initializes and returns a DeviceMesh for distributing computation across devices.

    To load the model with the weights and tensors distributed across TPUs, we first create a new DeviceMesh.
        - DeviceMesh represents a collection of hardware devices configured for distributed computation.
        - DeviceMesh was introduced in Keras 3 as part of the unified distribution API.

    The distribution API enables data and model parallelism.
        - This allows for efficient scaling of deep learning models on multiple accelerators and hosts.
        - The API leverages the underlying framework (e.g. JAX) to distribute the program and tensors according to the sharding directives.
            - This is done through a procedure called single program, multiple data (SPMD) expansion.
            - Check out more details in the new Keras 3 distribution API guide.
                --> https://keras.io/guides/distribution/

    Args:
        shape: A tuple specifying the shape of the overall `DeviceMesh`
            - `(8,)` for a data parallel only distribution,
            - `(4, 2)` for a model+data parallel distribution.
        batch_axis_name: A string indicating the axis name for the batch axis for DeviceMesh
        model_axis_name: The logical name of the model axis for the `DeviceMesh`

    Returns:
        A configured DeviceMesh instance.
            - Defaults to (1, 8) shape so that the weights are sharded across all 8 TPUs (v3-8).
        NOTE: This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`
            - i.e. It represents the computation devices in the global context.
    """
    return keras.distribution.DeviceMesh(
        shape=shape,
        axis_names=[batch_axis_name, model_axis_name],
        devices=jax.devices()
    )


def configure_layout_map(
        device_mesh: keras.distribution.DeviceMesh,
        model_axis_name: str = "model"
) -> keras.distribution.LayoutMap:
    """Configures and returns a LayoutMap for model weight distribution.

    LayoutMap from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys.
        - For example: 'token_embedding/embeddings' below, which are treated like regex to match tensor paths.
        - Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated.

    Args:
        device_mesh: The `DeviceMesh` that is used to populate the `TensorLayout.device_mesh`
        axis_name: The logical name of the model axis for the `DeviceMesh`

    Returns:
        A LayoutMap instance with predefined sharding configurations.
    """

    # A dict-like object that maps string to `TensorLayout` instances.
    layout_map = keras.distribution.LayoutMap(device_mesh)

    # Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
    layout_map["token_embedding/embeddings"] = (None, model_axis_name)

    # Regex to match against the query, key and value matrices in the decoder attention layers
    layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (None, model_axis_name, None)

    # etc.
    layout_map["decoder_block.*attention_output.*kernel"] = (None, None, model_axis_name)
    layout_map["decoder_block.*ffw_gating.*kernel"] = (model_axis_name, None)
    layout_map["decoder_block.*ffw_linear.*kernel"] = (None, model_axis_name)

    return layout_map


def update_distribution_strategy(
        device_mesh: keras.distribution.DeviceMesh,
        layout_map: keras.distribution.LayoutMap
) -> None:
    """Loads a distributed Gemma model based on the provided device mesh and layout map.

    ModelParallel allows you to shard model weights or activation tensors across all devcies on the DeviceMesh.
    In this case, some of the Gemma 7B model weights are sharded across 8 TPU chips according the layout_map.

    Args:
        model_name: The name of the Gemma model to load.
        device_mesh: A DeviceMesh instance for model distribution.
        layout_map: A LayoutMap instance defining how to distribute model weights.

    Returns:
        None; The keras backend is updated with the appropriate distribution strategy
    """

    # Shard across devices in the mesh and update distribution strategy accordingly
    model_parallel = keras.distribution.ModelParallel(device_mesh, layout_map, batch_dim_name="batch")
    keras.distribution.set_distribution(model_parallel)


def get_distributed_gemma(model_name: str) -> keras_nlp.models.GemmaCausalLM:
    """Obtain the TPU compatible Gemma model of your choice.

    Args:
        model_name: The name of the Gemma model to load. One of:
            - 'gemma_2b_en'
            - 'gemma_7b_en'
            - 'gemma_instruct_2b_en'
            - 'gemma_instruct_7b_en'

    Returns:
        A loaded Gemma model instance configured for distributed computation.
    """
    # Return the model
    return keras_nlp.models.GemmaCausalLM.from_preset(model_name)


def do_gemma_prep(return_device_mesh=True, return_layout_map=True):
    """Does the necessary steps so that we can instantiate a model properly

    Args:
        return_*: Whether to return the respective object as part of a dictionary
            - The key is the name (*) and the value is the object itself
    """
    device_mesh = initialize_device_mesh()
    layout_map = configure_layout_map(device_mesh)
    update_distribution_strategy(device_mesh, layout_map)

    return_map = {}
    if not (return_device_mesh or return_layout_map):
        pass
    else:
        if return_device_mesh:
            return_map["device_mesh"] = device_mesh
        if return_layout_map:
            return_map["layout_map"] = layout_map
    return return_map


setup_objects = do_gemma_prep()
model = get_distributed_gemma("gemma_instruct_7b_en")
GLOBAL_TOKENIZER = keras_nlp.models.GemmaTokenizer.from_preset("gemma_instruct_7b_en")

print(f"SETUP ARTIFACTS: \n{setup_objects}\n\nMODEL OBJECT:\n{model}")
print("\n\n\n... MODEL SUMMARY ...\n\n")
model.summary()


# %% [markdown]
# <br>
#
# **Double check how things are distributed/named**

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:28:18.397954Z","iopub.execute_input":"2024-04-10T15:28:18.398285Z","iopub.status.idle":"2024-04-10T15:28:18.404153Z","shell.execute_reply.started":"2024-04-10T15:28:18.398256Z","shell.execute_reply":"2024-04-10T15:28:18.403193Z"}}
def see_layer_info(model, layer_name='decoder_block_1'):
    block = model.backbone.get_layer(layer_name)
    print(f"{'-' * 70}")
    print(f"\n... INFO FOR LAYER: {layer_name} ...\n")
    print(f"{'-' * 70}")
    print(f"\tLAYER TYPE --> {type(block)}")
    print(f"{'-' * 70}")
    for variable in decoder_block_1.weights:
        print(
            f'\n\tNAME: {variable.path:<58}\n\tSHAPE: {str(variable.shape):<16}\n\tSHARDING SPECIFICATION: {str(variable.value.sharding.spec)}\n')
    print(f"{'-' * 70}")


# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:28:18.405949Z","iopub.execute_input":"2024-04-10T15:28:18.406257Z","iopub.status.idle":"2024-04-10T15:28:47.904488Z","shell.execute_reply.started":"2024-04-10T15:28:18.406229Z","shell.execute_reply":"2024-04-10T15:28:47.903239Z"}}
# Check that it works...
model.generate(["Best comedy movies in the 90s ", 'hello'], max_length=64)

# %% [markdown]
# <br>
#
# <b>As per unsloth</b>

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:39:55.321365Z","iopub.execute_input":"2024-04-10T15:39:55.321885Z","iopub.status.idle":"2024-04-10T15:39:55.32811Z","shell.execute_reply.started":"2024-04-10T15:39:55.321849Z","shell.execute_reply":"2024-04-10T15:39:55.326766Z"}}
model.preprocessor.add_start_token = True
model.preprocessor.end_token = False

# %% [markdown]
# <b> Timing Check With Longer Prompt


# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:40:01.512752Z","iopub.execute_input":"2024-04-10T15:40:01.51308Z","iopub.status.idle":"2024-04-10T15:40:02.000347Z","shell.execute_reply.started":"2024-04-10T15:40:01.513048Z","shell.execute_reply":"2024-04-10T15:40:01.999356Z"}}
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

results = []

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:40:27.378717Z","iopub.execute_input":"2024-04-10T15:40:27.379197Z","iopub.status.idle":"2024-04-10T15:40:31.591199Z","shell.execute_reply.started":"2024-04-10T15:40:27.379157Z","shell.execute_reply":"2024-04-10T15:40:31.590131Z"}}
texts = pd.read_parquet('/kaggle/input/newprompttext/new_original_data.parquet').values.ravel().tolist()
prompts = pd.read_csv('/kaggle/input/newprompttext/chatgptdeepseek_newprompt.csv').values.ravel().tolist()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:41:07.220704Z","iopub.execute_input":"2024-04-10T15:41:07.221126Z","iopub.status.idle":"2024-04-10T15:41:11.012647Z","shell.execute_reply.started":"2024-04-10T15:41:07.221095Z","shell.execute_reply":"2024-04-10T15:41:11.011505Z"}}
texts = [np.random.choice(item.split('\n\n')) for item in texts]

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:47:55.749438Z","iopub.execute_input":"2024-04-10T15:47:55.750457Z","iopub.status.idle":"2024-04-10T15:47:55.765465Z","shell.execute_reply.started":"2024-04-10T15:47:55.750411Z","shell.execute_reply":"2024-04-10T15:47:55.764576Z"}}
texts = [item for item in texts if len(item) > 64]

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:49:07.007089Z","iopub.execute_input":"2024-04-10T15:49:07.007545Z","iopub.status.idle":"2024-04-10T15:49:07.013593Z","shell.execute_reply.started":"2024-04-10T15:49:07.007509Z","shell.execute_reply":"2024-04-10T15:49:07.012533Z"}}
len(texts), texts[:5]

# %% [code] {"execution":{"iopub.status.busy":"2024-04-10T15:49:09.551252Z","iopub.execute_input":"2024-04-10T15:49:09.551621Z","iopub.status.idle":"2024-04-10T15:56:45.445145Z","shell.execute_reply.started":"2024-04-10T15:49:09.551592Z","shell.execute_reply":"2024-04-10T15:56:45.443776Z"}}
import time

seed = int(time.time()) % 1231435
# seed = 52362624
np.random.seed(seed)

results = []

for loop in tqdm(range(1000000)):
    L = []

    for _ in range(1):
        text = texts[np.random.choice(len(texts))]
        prompt = prompts[np.random.choice(len(prompts))]

        PROMPT = f'''{prompt[:-1] + ':'}\n"""{text}"""'''
        darien_prompt_template_str = f'''<bos><start_of_turn>user
                {PROMPT}<end_of_turn>
                <start_of_turn>model
                Rewritten Text:'''

        L.append({
            'touse': darien_prompt_template_str,
            'original_text': text,
            'prompt': prompt
        })

    output_text = model.generate([item['touse'] for item in L],
                                 max_length=1024,
                                 )
    for result, ddd in zip(output_text, L):
        results.append(
            {
                'original_text': ddd['original_text'],
                'prompt': ddd['prompt'],
                'rewrite_prompt': result,
            }
        )

    if loop % 10 == 1:
        df_result = pd.DataFrame(results)
        df_result.to_csv(f'./generated_text_prompt_tpu.csv')

