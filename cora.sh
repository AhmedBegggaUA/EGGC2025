#!/bin/bash

# Array de métodos y valores de renyi_loss
#methods=("mincut" "diffpool" "dmonpool")
methods=("mincut")
renyi_values=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#renyi_values=(0.3)
# Crear array con todas las combinaciones
experiments=()
for method in "${methods[@]}"; do
    for renyi in "${renyi_values[@]}"; do
        experiments+=("$method $renyi")
    done
done

# Función para lanzar un grupo de experimentos
launch_group() {
    local start=$1
    local end=$2
    
    for i in $(seq $start $end); do
        if [ $i -lt ${#experiments[@]} ]; then
            read method renyi <<< "${experiments[$i]}"
            echo "Lanzando experimento con method=$method, renyi_loss=$renyi"
            python main.py \
                --dataset Cora \
                --method "$method" \
                --renyi_loss "$renyi" \
                --log_file "./log/cora/log_${method}_${renyi}.txt" &
        fi
    done
}

# Lanzar experimentos en grupos de 8
total=${#experiments[@]}
for ((i=0; i<total; i+=11)); do
    echo "Lanzando grupo de experimentos $((i/11 + 1))"
    launch_group $i $((i+10))
    sleep 2  # Esperar 5 segundos antes del siguiente grupo
    wait     # Esperar a que termine el grupo actual antes de lanzar el siguiente
done

# Esperar a que terminen todos los procesos
wait

echo "Todos los experimentos han sido completados"