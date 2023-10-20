clc
clear all
close all
warning off all
close all
hold on

c1 = [0 0 0; 1 0 0; 1 0 1; 1 1 0];
c2 = [0 0 1; 0 1 1; 1 1 1; 0 1 0];

n_clases = ["c1", "c2"]



colors = ["blue", "yellow", "magenta", "cyan", "white", "black", "red", "green"];

scatter3(c1(:,1), c1(:,2), c1(:,3), 100, 'r', 'filled');
scatter3(c2(:,1), c2(:,2), c2(:,3), 100, 'g', 'filled');

legend('Class 1', 'Class 2');

view(3);
title("Ejercicio 1")
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
hold on
axis square;
grid on;

flag = 1;
fc = 1;
classes_count = 2; % Número de clases en el problema

while flag == 1
           
    msj = 'Ingrese el nuevo vector de la forma -> [x y z]: ';
    new_vector = input(msj);
    
    if all(new_vector >= 0) && all(new_vector <= 1)
        hold on;
        scatter3(new_vector(1), new_vector(2), new_vector(3), 100, colors(fc), 'filled');
        fc = fc + 1;
        hold off;
        legend('Class 1', 'Class 2', 'Vector Us');
        fprintf('El punto está dentro del cubo.\n');
    
        x_vec = [new_vector(1) new_vector(2) new_vector(3)];
        
        c1_avg = get_average(c1);
        c2_avg = get_average(c2);   
        
        c1_minus_avg = c1 - c1_avg;
        c2_minus_avg = c2 - c2_avg;
        
        x_vec_minus_c1_avg = x_vec - c1_avg;
        x_vec_minus_c2_avg = x_vec - c2_avg;
        
        c1_tot = get_tot_matrix(c1_minus_avg);
        c2_tot = get_tot_matrix(c2_minus_avg);
        
        dist1 = x_vec_minus_c1_avg * c1_tot * x_vec_minus_c1_avg';
        dist2 = x_vec_minus_c2_avg * c2_tot * x_vec_minus_c2_avg';
        
        fprintf('Distancia Mahalanobis a la Clase 1: %f\n', dist1);
        fprintf('Distancia Mahalanobis a la Clase 2: %f\n', dist2);

        disp('El punto pertenece a la clase:')
        if dist1 < dist2
            disp('1')
        else
            disp('2')
        end
        hold on

        fprintf("Aplicando Max Probabilidad\n");

        [clase, prob] = get_class_and_max_prob(c1, c2, dist1, dist2, x_vec, n_clases);
        
        fprintf('Clase predicha por Max Probabilidad: %d\n', clase);
        fprintf('Probabilidad Max Probabilidad: %f\n', prob*100);
    
    else
        fprintf('El punto está fuera del cubo.\n');
    end
end

function average = get_average(input_vec)
    [rows, cols] = size(input_vec);
    average = zeros(1, cols);

    for i = 1:cols
        current_sum = 0;
        for j = 1:rows
            current_sum = current_sum + input_vec(j, i);
        end
        average(i) = current_sum / rows;
    end
end

function tot_mat = get_tot_matrix(input_mat)
    input_mat_tp = input_mat';
    [rows, cols] = size(input_mat);

    tot_mat = 1/rows * input_mat_tp * input_mat;
end

function [class_no, max_likelihood] = get_class_and_max_prob(c1, c2, dist1, dist2, new_vector, classes_count)
    max_likelihood = -inf;
    class_no = -1;
    
    k = length(new_vector); 
    
    for class_index = 1:size(classes_count)
        if class_index == 1
            Sigma = get_tot_matrix(c1 - get_average(c1));
            delta = new_vector - get_average(c1);

        else
            Sigma = get_tot_matrix(c2 - get_average(c2));
            delta = new_vector - get_average(c2);
        end
        
        likelihood = 0;
        likelihood = (1 / ((2*pi)^(k/2) * sqrt(det(Sigma)))) * exp(-0.5 * delta * inv(Sigma) * delta');
        
        if likelihood > max_likelihood
            max_likelihood = likelihood;
            class_no = class_index;
        end
    end
end
