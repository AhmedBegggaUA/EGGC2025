#!/bin/bash

# Array de métodos y valores de renyi_loss
methods=("mincut" "diffpool" "dmonpool")
renyi_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

total_experiments=$((${#methods[@]} * ${#renyi_values[@]}))
current=0

# Ejecutar experimentos secuencialmente
for method in "${methods[@]}"; do
    for renyi in "${renyi_values[@]}"; do
        ((current++))
        echo "Lanzando experimento $current de $total_experiments con method=$method, renyi_loss=$renyi"
        python main.py \
            --dataset PubMed \
            --method "$method" \
            --renyi_loss "$renyi" \
            --device "cuda:1" \
            --log_file "./log/pubmed/log_${method}_${renyi}.txt"
        
        echo "Experimento $current completado. Faltan $((total_experiments - current))"
    done
done

echo "Todos los experimentos han sido completados"