clc
clear all
close all
warning off all
close all
hold on

c1 = [0 0 0; 1 0 0; 1 0 1; 1 1 0];
c2 = [0 0 1; 0 1 1; 1 1 1; 0 1 0];

colors = [ "blue", "yellow", "magenta", "cyan", "white", "black","red", "green"];

scatter3(c1(:,1), c1(:,2), c1(:,3), 100, 'r', 'filled');
scatter3(c2(:,1), c2(:,2), c2(:,3), 100, 'g', 'filled');

legend('Class 1', 'Class 2');

view(3);
% axis([-1,2,-1,2,-1,2]);
title("Ejercicio 1")
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
hold on
axis square;
grid on;


flag = 1;
fc = 1;
while flag == 1
           
    msj = 'Ingrese el nuevo vector de la forma -> [x y z]): ';
    new_vector = input(msj);
    
    if all(new_vector >= 0) && all(new_vector <= 1)
        hold on;
        scatter3(new_vector(1), new_vector(2), new_vector(3), 100, colors(fc), 'filled');
        fc = fc +1;
        hold off;
        legend('Class 1', 'Class 2', 'Vector Us');
        fprintf('El punto está dentro del cubo.\n');
    
        x_vec = [new_vector(1) new_vector(2) new_vector(3)]
        
        c1_avg = get_average(c1)
        % c1_mean = mean(c1)
        c2_avg = get_average(c2);   
        
        c1_minus_avg = c1 - c1_avg;
        c2_minus_avg = c2 - c2_avg;
        
        x_vec_minus_c1_avg = x_vec - c1_avg;
        x_vec_minus_c2_avg = x_vec - c2_avg;
        
        c1_tot = get_tot_matrix(c1_minus_avg);
        c2_tot = get_tot_matrix(c2_minus_avg);
        
        dist1 = x_vec_minus_c1_avg * c1_tot * x_vec_minus_c1_avg'
        dist2 = x_vec_minus_c2_avg * c2_tot * x_vec_minus_c2_avg'
        disp('El punto pertenece a  la clase:')
        if dist1 < dist2
            disp('1')
        else
            disp('2')
        end
        hold on
    
    else
        fprintf('El punto está fuera del cubo.\n');
    end
end



%Funcion para miu
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

%Funcion distancia total_1
function tot_mat = get_tot_matrix(input_mat)
    input_mat_tp = input_mat';
    [rows, cols] = size(input_mat);

    tot_mat = 1/rows * input_mat_tp * input_mat;
end