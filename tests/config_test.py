import yaml
from edge_llm.config import AppConfig
with open('models.yaml', 'r') as f:
    config = AppConfig(**yaml.safe_load(f))
print('✓ Config loaded successfully')
print(f'Found models: {[m.id for m in config.models]}')
