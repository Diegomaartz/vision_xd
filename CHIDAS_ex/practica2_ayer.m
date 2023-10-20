clc
clear
close all
warning('off','all');

a = imread('frutas.png');
figure(1)
imshow(a)

figure(2)
clases = impixel(a);
clase1 = clases(1:3,:);
clase2 = clases(4:6,:);
clase3 = clases(7:9,:);
vector = clases(10,:)';

figure(3)
plot3(clase1(1,:), clase1(2,:), clase1(3,:), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r')
hold on
plot3(clase2(1,:), clase2(2,:), clase2(3,:), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b')
plot3(clase3(1,:), clase3(2,:), clase3(3,:), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g')
plot3(vector(1), vector(2), vector(3), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k')
grid on
legend('clase1', 'clase2', 'clase3','vector')
xlabel("X")
ylabel("Y")
disp(clase1)
disp('Fin del programa')

x_vec = vector;
no_of_classes = 3;
elements_p_class = 3;

distances = zeros(no_of_classes,1);


[class_no, min_dist] = get_class_and_min_dist(clases, no_of_classes, elements_p_class, x_vec);

fprintf("\nEl vector pertenece a la clase: %d\n", class_no)
fprintf("A una distancia de: %f\n", min_dist)

function [class_no, current_min] = get_class_and_min_dist(classes_mat, classes_count, elements_p_class, vector)
    current_min = inf;
    class_no = -1;

    for class_index = 1:classes_count
        start_row = elements_p_class * (class_index - 1) + 1;
        end_row = class_index * elements_p_class;

        class_values = classes_mat(start_row:end_row, :);

        class_avg = mean(class_values); % Calculamos la media de cada clase
        class_minus_avg = class_values - class_avg;

        x_vec_minus_class_avg = vector - class_avg;

        class_tot = get_tot_matrix(class_minus_avg); % Calculamos la matriz de covarianza de cada clase

        dist = x_vec_minus_class_avg * class_tot * x_vec_minus_class_avg';

        if dist < current_min
            current_min = dist;
            class_no = class_index;
        end

    end
end

function tot_mat = get_tot_matrix(input_mat)
    input_mat_tp = input_mat';
    [rows, ~] = size(input_mat);

    tot_mat = 1/rows * input_mat_tp * input_mat;
end