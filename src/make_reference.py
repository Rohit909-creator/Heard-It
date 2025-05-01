import torch
import torch.nn as nn
from Preprocessing import load_and_preprocess_audio_file
from Trainer import ResNetMel, ResNetMelLite
from Utils import EnhancedSimilarityMatcher, Matcher
import json

with open("./mswc_cache/classes.txt", 'r') as f:
    s = f.read()
   
# with open("./AI_audios_cache/classes.txt", 'r') as f:
#     s = f.read()
classes = s.split("\n")
num_classes = len(classes)

# checkpoint_path = "./lightning_logs/version_23/checkpoints/epoch=14-step=46560.ckpt"
checkpoint_path = "./lightning_logs/version_27/checkpoints/epoch=9-step=31040.ckpt"
# checkpoint_path = "./lightning_logs/version_25/checkpoints/epoch=13-step=44142.ckpt"
# checkpoint_path = "./lightning_logs/version_26/checkpoints/epoch=24-step=78825.ckpt"
# Wakeword only Dataset
# model = ResNetMel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model = ResNetMelLite.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
model.eval()

mel_spec1 = load_and_preprocess_audio_file("./Audios4testing/shambu_1.wav", max_duration=1.0)

mel_spec_tensor = torch.tensor([mel_spec1], dtype=torch.float32)
mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
mel_spec_tensor = mel_spec_tensor.cuda()
print(mel_spec_tensor.shape)
# model.model.fc[1] = nn.Sequential()

model.model.fc[4] = nn.Sequential()
print(model.named_modules)
out = model(mel_spec_tensor)
# Save the embeddings to a JSON file
data = {"embeddings": out.cpu().detach().numpy().tolist()}
with open("Shambu_27thModel_epoch9.json", "w") as f:
    f.write(json.dumps(data))
print("saved embeddings")