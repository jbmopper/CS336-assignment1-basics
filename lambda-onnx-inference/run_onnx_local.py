import numpy
import onnxruntime as rt

session = rt.InferenceSession("./export/models/baseline_best.onnx", providers=["CPUExecutionProvider"])

print("onnx inputs")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("onnx outputs")
for outp in session.get_outputs():
    print(outp.name, outp.shape, outp.type)

