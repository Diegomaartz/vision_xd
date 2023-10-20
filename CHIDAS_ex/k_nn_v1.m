clc
clear
close all
warning off;

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'k'];

msj_no_classes = 'Ingresa el numero de clases a utilizar: ';
no_classes = input(msj_no_classes);

msj_no_elementos_p_clase = 'Ingresa el representantes por clase: ';
no_elementos = input(msj_no_elementos_p_clase);

no_elementos_dataset = no_elementos * no_classes;

classes_elements = zeros(no_elementos_dataset, 3);

playa = imread('frutas.png');

[rows, cols, ~] = size(playa);
    
imshow(playa);
hold on;

c_grav = zeros(length(no_classes), 2);
dataset_rgb = zeros(no_elementos_dataset, 3);
dataset_labels = zeros(no_elementos_dataset, 1);
counter = 1;

while counter <= no_classes
    fprintf('Captura el centro de gravedad para la clase %d:\n', counter);
    c_grav(counter, :) = ginput(1);
    c_grav_x = c_grav(counter, 1);
    c_grav_y = c_grav(counter,2);

    disp(c_grav_x)
    disp(c_grav_y)
    

    if ~point_is_in_image(c_grav_x, c_grav_y, cols, rows)
        fprintf('\n\nSelecciona solo puntos dentro de la imagen\n');
    else
        plot(c_grav_x, c_grav_y, 'o', 'MarkerSize', 15, 'MarkerFaceColor', 'black');
        
        %Obtiene los puntos n random en una matriz
        [x_coordinates, y_coordinates] = get_n_points_inside_image_limits(c_grav_x, c_grav_y, cols, rows, no_elementos);
        
        %Obtiene el RGB de los puntos anteriores
        [class_rgb_values, class_labels] = get_rgb_from_coordinates(playa, x_coordinates, y_coordinates, no_elementos, counter);
        
        start_idx = (counter - 1) * no_elementos + 1;
        end_idx = start_idx + no_elementos - 1;

        %Creamos un dataset de valores RGB y labels
        dataset_rgb(start_idx:end_idx, :) = class_rgb_values;
        dataset_labels(start_idx:end_idx) = class_labels;
        
        %Ploteamos los puntos
        color = colors(counter);
        scatter(x_coordinates, y_coordinates, color, "filled");
        
        counter = counter + 1;
    end
end
user_input = 's';

while strcmp(user_input, 's')

        rgvector_knn = [0 0 0];

        predicted_lass = -1;

        msg_vector = fprintf("Selecciona el vector en la imagen: ");
        vector_knn = ginput(1)
        [rgb_value_knn] = impixel(playa,vector_knn(1), vector_knn(2));

        msg_select_k_value = "Teclea el número K para el cálculo del KNN (K > 0 e IMPAR): ";
        k_for_knn = input(msg_select_k_value);
         disp(mod(k_for_knn,2));
        while mod(k_for_knn,2) == 0 && k_for_knn <= 0 
            k_for_knn = input(msg_select_k_value);
        end
        predicted_class = knn_euclidean(dataset_rgb, dataset_labels, k_for_knn, rgb_value_knn);
        disp("Pertenece a la clase: " + predicted_class + " , color: " + colors(predicted_class));        
end
    
function [class_no] = knn_euclidean(dataset_rgb, dataset_labels, k, vector)
    disp(vector)
    distances = zeros(size(dataset_rgb, 1), 1);
    
    for i = 1:size(dataset_rgb, 1)
        distances(i) = norm(vector - dataset_rgb(i, :));
    end
    sortedIndices = 0;
    [~, sortedIndices] = sort(distances);
    disp(size(sortedIndices));
    nearestNeighbors = sortedIndices(1:k);
    
    neighborLabels = dataset_labels(nearestNeighbors);
    
    class_no = mode(neighborLabels);
end

%Plotear la nube de puntos
function [x_coordinates, y_coordinates] = get_n_points_inside_image_limits(c_grav_x, c_grav_y, img_size_x, img_size_y, elements_p_class)
    separated_factor = 30;
    x_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_x);
    y_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_y);

       %valida que esten dentro de la imagen, si se salian = filas o colm
    for i = 1 : elements_p_class
        x_value = x_coordinates(i);
        if x_value < 1
            x_value = 1;
        elseif x_value > img_size_x
            x_value = img_size_x;
        end

        y_value = y_coordinates(i);
        if y_value < 1
            y_value = 1;
        elseif y_value > img_size_y
            y_value = img_size_y;
        end

        x_coordinates(i) = x_value;
        y_coordinates(i) = y_value;
    end
end

%Obtenemos el RGB de cada punto random generado, regresando una matriz rgb
%y de labels
function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end

%Funcion para checar si da click dentro de la imagen, recibe cg_x, cg_y, y
%tamaño de la imagen filas y columnas.
function point_in_image = point_is_in_image(x, y, img_size_x, img_size_y)
    if  (x >= 1 && y >= 1) && (x <= img_size_x && y <= img_size_y)
        point_in_image = true;
        return
    end

    point_in_image = false;
end
