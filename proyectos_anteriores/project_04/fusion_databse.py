import json
from datetime import datetime

# Cargar los dos archivos
with open("aux_database_1.json", "r") as f:
    data1 = json.load(f)

with open("aux_database_2.json", "r") as f:
    data2 = json.load(f)

# Unir las dos listas
combined = data1 + data2

# Ordenar por start_time
combined.sort(key=lambda x: datetime.fromisoformat(x["start_time"].replace("Z", "+00:00")))

# Guardar el resultado
with open("aux_database.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Total de registros: {len(combined)}")