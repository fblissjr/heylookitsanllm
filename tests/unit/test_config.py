import tomllib
from heylook_llm.config import AppConfig
with open('models.toml', 'rb') as f:
    config = AppConfig(**tomllib.load(f))
print('Config loaded successfully')
print(f'Found models: {[m.id for m in config.models]}')
