X = [0 0 0; 1 0 0; 1 1 0; 1 0 1; 0 1 0; 0 1 1; 0 0 1; 1 1 1];
y = [0 0 0 0 1 1 1 1];

w = zeros(1, 4);
for i = 1:3
    w(i) = input(['Ingrese el valor del peso w', num2str(i), ': ']);
end

w(4) = input('Ingresa el valor de w0: ');

r = input('Ingrese el valor del coeficiente de error r: ');
if r <= 0
    disp('El coeficiente de error debe ser mayor a 0.');
    return;
end

converge = false;
etapa = 0;

while ~converge
    converge = true;
    fprintf('Étapa %d:\n', etapa + 1);

    for i = 1:length(X)
        x = X(i,:);
        xn = [x, 1];
        fsal = perceptron(xn, w);

        if fsal >= 0 && y(i) == 0
            fprintf('Para entrada [%d, %d, %d], pertenece a la clase 1\n', x);
            w = w - r * xn;  % C1
            converge = false;
        elseif fsal <= 0 && y(i) == 1
            fprintf('Para entrada [%d, %d, %d], pertenece a la clase 2\n', x);
            w = w + r * xn;  % C2
            converge = false;
        else
            fprintf('Para entrada [%d, %d, %d], permanece sin cambios\n', x);
        end
    end
    etapa = etapa + 1;
end

fprintf('\nEl perceptrón ha convergido:\n');
fprintf('Valores finales de los pesos: %s\n', mat2str(w));

figure;
axis = axes('Position', [0.1, 0.1, 0.8, 0.8], 'Box', 'on', 'GridLineStyle', '--');
hold on;

for i = 1:length(X)
    if y(i) == 0
        scatter3(X(i,1), X(i,2), X(i,3), 'r', 'o', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Clase 1');
    else
        scatter3(X(i,1), X(i,2), X(i,3), 'b', 'x', 'LineWidth', 2, 'DisplayName', 'Clase 2');
    end
end

xlabel('X');
ylabel('Y');
zlabel('Z');

[xx, yy] = meshgrid(0:1, 0:1);
zz = (-w(1) * xx - w(2) * yy - w(4)) / w(3);
surf(xx, yy, zz, 'FaceAlpha', 0.5);

legend('Location', 'Best');
legend('show');

text(0, 0, 0, 'Clase 1', 'Color', 'c');
text(1, 0, 0, 'Clase 1', 'Color', 'c');
text(1, 1, 0, 'Clase 1', 'Color', 'c');
text(1, 0, 1, 'Clase 1', 'Color', 'c');
text(0, 1, 0, 'Clase 2', 'Color', 'm');
text(0, 1, 1, 'Clase 2', 'Color', 'm');
text(0, 0, 1, 'Clase 2', 'Color', 'm');
text(1, 1, 1, 'Clase 2', 'Color', 'm');



hold off;
axis equal;
view(3);

function fsal = perceptron(x, w)
    fsal = dot(x, w);
end
