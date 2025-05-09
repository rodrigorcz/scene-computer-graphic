''' 
Projeto 1 - Computação Grafica

Aluno: Rodrigo Rodrigues de Castro (13695362)

Descrição da Cena: 
    Pique Nique, onde o tempo vai passando e com isso o sol vai se movendo, 
    as flores crescendo e as estrelas girando

Comandos:

<- (Seta Esquerda): Move o Sol para a esquerda (translação no eixo X) 
-> (Seta Direita): Move o Sol para a direita (translação no eixo X) 
^ (Seta Cima): Rotaciona as estrelas no sentido horário
v (Seta Baixo): Rotaciona as estrelas no sentido anti-horário
Z: Aumenta a escala das flores
X: Diminui a escala das flores
T: Aplica uma transformação combinada:
    Move o Sol para a esquerda,
    Rotaciona as estrelas no sentido horário,
    Aumenta levemente a escala das flores.

P: Ativa o modo malha poligonal (wireframe)
F: Retorna ao modo de renderização normal (sólido/preenchido)

Programa feito utilizando Python 3.13.0

'''
import glfw
from OpenGL.GL import *
import numpy as np
import glm
import math
import random

# Iniciando a janela
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
window = glfw.create_window(700, 700, "Programa", None, None)

if (window == None):
    print("Failed to create GLFW window")
    glfwTerminate()
    
glfw.make_context_current(window)

# Shaders
vertex_code = """
        attribute vec3 position;
        uniform mat4 mat_transformation;
        void main(){
            gl_Position = mat_transformation * vec4(position,1.0);
        }
        """

fragment_code = """
        uniform vec4 color;
        void main(){
            gl_FragColor = color;
        }
        """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)


# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")


glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)


# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)


# -------------- Criação dos Vertices --------------------------
PI = 3.141592

# Função para criar os vertices de uma esfera
def criar_esfera(r, num_sectors=24, num_stacks=24):
    PI = np.pi
    sector_step = (PI * 2) / num_sectors
    stack_step = PI / num_stacks

    def F(u, v, r):
        return np.array([
            r * np.sin(v) * np.cos(u),
            r * np.sin(v) * np.sin(u),
            r * np.cos(v)
        ], dtype=np.float32)

    vertices_list = []
    for i in range(0,num_sectors): # para cada sector (longitude)
        for j in range(0,num_stacks): # para cada stack (latitude)
            
            u = i * sector_step # angulo setor
            v = j * stack_step # angulo stack
            
            un = 0 # angulo do proximo sector
            if i+1==num_sectors:
                un = PI*2
            else: un = (i+1)*sector_step
                
            vn = 0 # angulo do proximo stack
            if j+1==num_stacks:
                vn = PI
            else: vn = (j+1)*stack_step
            
            # vertices do poligono
            p0=F(u, v, r)
            p1=F(u, vn, r)
            p2=F(un, v, r)
            p3=F(un, vn, r)
            
            # triangulo 1 (primeira parte do poligono)
            vertices_list.append(p0)
            vertices_list.append(p2)
            vertices_list.append(p1)
            
            # triangulo 2 (segunda e ultima parte do poligono)
            vertices_list.append(p3)
            vertices_list.append(p1)
            vertices_list.append(p2)

    return np.array(vertices_list, dtype=np.float32)

# Função para criar os vertices de um pao 3D
def criar_pao(rx=1.0, ry=1.5, rz=0.8):
    vertices = np.zeros(24, [("position", np.float32, 3)])

    # As dimensões agora são diferentes para cada eixo
    vertices['position'] = [
        # Face frontal
        (-rx, -ry, +rz),
        (+rx, -ry, +rz),
        (-rx, +ry, +rz),
        (+rx, +ry, +rz),

        # Direita
        (+rx, -ry, +rz),
        (+rx, -ry, -rz),
        (+rx, +ry, +rz),
        (+rx, +ry, -rz),

        # Traseira
        (+rx, -ry, -rz),
        (-rx, -ry, -rz),
        (+rx, +ry, -rz),
        (-rx, +ry, -rz),

        # Esquerda
        (-rx, -ry, -rz),
        (-rx, -ry, +rz),
        (-rx, +ry, -rz),
        (-rx, +ry, +rz),

        # Base
        (-rx, -ry, -rz),
        (+rx, -ry, -rz),
        (-rx, -ry, +rz),
        (+rx, -ry, +rz),

        # Topo (achatado por enquanto)
        (-rx, +ry, +rz),
        (+rx, +ry, +rz),
        (-rx, +ry, -rz),
        (+rx, +ry, -rz)
    ]

    return vertices['position']

# Função para criar os vertices de um retangulo 2D
def criar_retangulo(largura=2.0, comprimento=2.0):
    # Retorna apenas vértices (sem cores)
    vertices = [
       
        (-largura/2, 0.0, -comprimento/2),  # V0
        (0.0, 0.0, -comprimento/2),         # V1
        (-largura/2, 0.0, 0.0),             # V2
        

        (0.0, 0.0, -comprimento/2),         # V1
        (0.0, 0.0, 0.0),                    # V3
        (-largura/2, 0.0, 0.0),             # V2
        
        (0.0, 0.0, -comprimento/2),         # V1
        (largura/2, 0.0, -comprimento/2),   # V4
        (0.0, 0.0, 0.0),                    # V3
        
        (largura/2, 0.0, -comprimento/2),   # V4
        (largura/2, 0.0, 0.0),              # V5
        (0.0, 0.0, 0.0)                     # V3
    ]
    return np.array(vertices, dtype=np.float32)

# Função para criar os vertices de uma piramide de base quadrada 3D
def criar_piramide(comprimento):

    vertices= [
        # Face 1 
        (0.0, comprimento*1.5, 0),
        (comprimento/2, 0.0, -comprimento/2),
        (comprimento/2, 0.0, +comprimento/2),

        # Face 2 
        (0.0, comprimento*1.5, 0),
        (comprimento/2, 0.0, +comprimento/2),
        (-comprimento/2, 0.0, +comprimento/2),

        # Face 3
        (0.0, comprimento*1.5, 0),
        (-comprimento/2, 0.0, +comprimento/2),
        (-comprimento/2, 0.0, -comprimento/2),
        
        # Face 4 
        (0.0, comprimento*1.5, 0),
        (-comprimento/2, 0.0, -comprimento/2),
        (comprimento/2, 0.0, -comprimento/2),
        
        # Face 5 (quadrado)
        (comprimento/2, 0.0, +comprimento/2),
        (comprimento/2, 0.0, -comprimento/2),
        (-comprimento/2, 0.0, +comprimento/2),
        (-comprimento/2, 0.0, -comprimento/2),
    ]

    return np.array(vertices, dtype=np.float32)

# Função para criar os vertices de um lona 2D (retangulo inclinado)
def criar_lona(largura=2.0, profundidade=2.0, deslocamento=0.5):

    # Parte da frente (mais próxima)
    frente_esq = (-largura / 2, 0.0, 0.0)
    frente_dir = (largura / 2, 0.0, 0.0)

    # Parte do fundo (mesma largura, mas deslocada no X)
    fundo_esq = (-largura / 2 + deslocamento, 0.0, -profundidade)
    fundo_dir = (largura / 2 + deslocamento, 0.0, -profundidade)

    vertices = [
        fundo_esq, frente_esq, fundo_dir,
        frente_esq, frente_dir, fundo_dir
    ]
    return np.array(vertices, dtype=np.float32)

# Função para criar os vertices de um cilindro 3D
def criar_cilindro(r = 0.1, num_sectors = 20, num_stacks = 20, H = 0.9):

    # grid sectos vs stacks (longitude vs latitude)
    sector_step = (PI*2)/num_sectors # variar de 0 até 2π
    stack_step = H/num_stacks # variar de 0 até H

    # Angulo de t, altura h, raio r
    def CoordCilindro(t, h, r):
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = h
        return (x,y,z)

    vertices_list = []
    for j in range(0,num_stacks): # para cada stack (latitude)
        
        for i in range(0,num_sectors): # para cada sector (longitude) 
            
            u = i * sector_step # angulo setor
            v = j * stack_step # altura da stack
            
            un = 0 # angulo do proximo sector
            if i+1==num_sectors:
                un = PI*2
            else: un = (i+1)*sector_step
                
            vn = 0 # altura da proxima stack
            if j+1==num_stacks:
                vn = H
            else: vn = (j+1)*stack_step
            
            # verticies do poligono
            p0=CoordCilindro(u, v, r)
            p1=CoordCilindro(u, vn, r)
            p2=CoordCilindro(un, v, r)
            p3=CoordCilindro(un, vn, r)
            
            # triangulo 1 (primeira parte do poligono)
            vertices_list.append(p0)
            vertices_list.append(p2)
            vertices_list.append(p1)
            
            # triangulo 2 (segunda e ultima parte do poligono)
            vertices_list.append(p3)
            vertices_list.append(p1)
            vertices_list.append(p2)
            
            if v == 0:
                vertices_list.append(p0)
                vertices_list.append(p2)
                vertices_list.append(CoordCilindro(0, v, 0))
            if vn == H:
                #faz um triangulo a partir do mesmo angulo u, mas com as alturas em h = vn
                vertices_list.append(p1)
                vertices_list.append(p3)
                vertices_list.append(CoordCilindro(0, vn, 0))

    return np.array(vertices_list, dtype=np.float32)

# Função para criar os vertices de uma estrela 2D
def criar_estrela(num_pontas=5, raio_externo=0.3, raio_interno=0.15):
    vertices = []

    angulo_total = 2 * math.pi
    passo = angulo_total / (num_pontas * 2)  
    pontos = []
    for i in range(num_pontas * 2):
        raio = raio_externo if i % 2 == 0 else raio_interno
        angulo = i * passo
        x = raio * math.cos(angulo)
        y = raio * math.sin(angulo)
        pontos.append((x, y, 0.0))

    # Gera os triângulos com o centro no (0, 0, 0)
    for i in range(len(pontos)):
        p1 = (0.0, 0.0, 0.0)
        p2 = pontos[i]
        p3 = pontos[(i + 1) % len(pontos)]
        vertices.extend([p1, p2, p3])

    return np.array(vertices, dtype=np.float32)

# Função para criar os vertices de uma flor 2D
def criar_flor(num_petalas=8, raio_interno=0.1, raio_externo=0.2):
    vertices = []

    # Raio para o miolo 
    angulo = 2 * np.pi / num_petalas
    centro = (0.0, 0.0, 0.0)

    # Pétalas 
    for i in range(num_petalas):
        ang1 = i * angulo
        ang2 = (i + 1) * angulo

        x1 = raio_interno * np.cos(ang1)
        y1 = raio_interno * np.sin(ang1)

        xm = raio_externo * np.cos(ang1 + angulo / 2)
        ym = raio_externo * np.sin(ang1 + angulo / 2)

        x2 = raio_interno * np.cos(ang2)
        y2 = raio_interno * np.sin(ang2)

        # Triângulo da pétala
        vertices.append((x1, y1, 0.0))
        vertices.append((xm, ym, 0.0))
        vertices.append((x2, y2, 0.0))

    # Miolo
    for i in range(num_petalas):
        ang1 = i * angulo
        ang2 = (i + 1) * angulo

        x1 = raio_interno * np.cos(ang1)
        y1 = raio_interno * np.sin(ang1)

        x2 = raio_interno * np.cos(ang2)
        y2 = raio_interno * np.sin(ang2)

        # Triângulo do miolo
        vertices.append(centro)
        vertices.append((x1, y1, 0.0))
        vertices.append((x2, y2, 0.0))

    return np.array(vertices, dtype=np.float32)


# -------------- Configurações Gerais --------------------------

# Função que configura um Vertex Buffer Object (VBO) para enviar os dados dos vértices para a GPU
def setup_VBO(vertices):
    # Request a buffer slot from GPU
    buffer_VBO = glGenBuffers(1)
    # Make this buffer the default one
    glBindBuffer(GL_ARRAY_BUFFER, buffer_VBO)

    # Upload data
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_VBO)

    stride = vertices.strides[0]
    offset = ctypes.c_void_p(0)

    loc = glGetAttribLocation(program, "position")
    glEnableVertexAttribArray(loc)

    glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)

# Obtém a localização da variável uniforme color no shader
loc_color = glGetUniformLocation(program, "color")

# Função para muiltiplicação de matrizes 4x4
def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c



# -------------- Desenho dos Objetos --------------------------

# Função que desenha uma esfera 3D com 2 cores
def desenha_esfera(offset, num_triangulos, tx, ty, tz, color1, color2):

    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transl)

    for triangle in range(num_triangulos):
        random.seed( triangle )
        num = random.randint(0, 1)

        if num == 0 :
            R, G, B = color1
        else:
            R, G, B = color2

        glUniform4f(loc_color, R, G, B, 1)
        glDrawArrays(GL_TRIANGLES, offset + triangle*3, 3)  

# Função que desenha um cilindro 3D com 2 cores
def desenha_cilindro(offset, num_triangulos, tx, ty, tz, color):

    theta_x = np.radians(-125)  # Rotaciona para alinhar com eixo Y
    theta_z = np.radians(0)   # Inclina para dar perspectiva

    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)

    # Matriz de translação
    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32).reshape((4, 4)) 

    # Matriz de rotação no eixo X 
    mat_rotation_x = np.array([
        1.0,  0.0,   0.0,  0.0, 
        0.0,  cos_x, -sin_x, 0.0, 
        0.0,  sin_x,  cos_x, 0.0, 
        0.0,  0.0,   0.0,  1.0
    ], dtype=np.float32).reshape((4, 4))  

    # Matriz de rotação no eixo Z 
    mat_rotation_y = np.array([
        cos_z,  0.0, sin_z, 0.0, 
        0.0,    1.0, 0.0, 0.0, 
        -sin_z, 0.0, cos_z, 0.0, 
        0.0,    0.0, 0.0, 1.0
    ], dtype=np.float32).reshape((4, 4)) 

    mat_transform = mat_transl @ mat_rotation_y @ mat_rotation_x

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)

    R, G, B = color

    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLES, offset, num_triangulos * 3) 

# Função que desenha um pão 3D
def desenha_pao(offset, tx, ty, tz, angulo_x, angulo_y):
    rad_x = np.radians(angulo_x)
    rad_y = np.radians(angulo_y)

    cos_x = np.cos(rad_x)
    sin_x = np.sin(rad_x)
    cos_y = np.cos(rad_y)
    sin_y = np.sin(rad_y)

    # Matriz de Translação
    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    # Matriz de Rotação X
    mat_rot_x = np.array([
        1.0, 0.0,    0.0,   0.0,
        0.0, cos_x, -sin_x, 0.0,
        0.0, sin_x,  cos_x, 0.0,
        0.0, 0.0,    0.0,   1.0
    ], dtype=np.float32)

    # Matriz de Rotação Y
    mat_rot_y = np.array([
        cos_y, 0.0, sin_y, 0.0,
        0.0,   1.0, 0.0,   0.0,
       -sin_y, 0.0, cos_y, 0.0,
        0.0,   0.0, 0.0,   1.0
    ], dtype=np.float32)

    # Matriz final de transformação
    mat_transform = multiplica_matriz(mat_rot_y, mat_rot_x)
    mat_transform = multiplica_matriz(mat_transl, mat_transform)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)

    # Cores por face 
    cores_faces = [
        (1.0, 0.8, 0.6),  
        (1.0, 0.75, 0.5),  
        (1.0, 0.7, 0.4),  
        (1.0, 0.65, 0.35), 
        (0.9, 0.6, 0.3),   
        (0.7, 0.4, 0.2)    #
    ]

    for i in range(6):
        R, G, B = cores_faces[i % len(cores_faces)]
        glUniform4f(loc_color, R, G, B, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, offset + i * 4, 4)
   

# Função que desenha um quadrilatero inclinado
def desenha_quadrilatero(offset, num_triangulos, tx, ty, tz, angulo_inclinacao=20.0, color=(0.5, 0.2, 0.3)):

    ang_rad = np.radians(angulo_inclinacao)
    cos_a = np.cos(ang_rad)
    sin_a = np.sin(ang_rad)

    # Matriz de rotação em X 
    mat_rotacao_x = np.array([
        1.0, 0.0, 0.0, 0.0,
        0.0, cos_a, -sin_a, 0.0,
        0.0, sin_a, cos_a, 0.0,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    # Matriz de translação 
    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    # Combina translação + rotação
    mat_transform = multiplica_matriz(mat_transl, mat_rotacao_x)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)

    R, G, B = color
    glUniform4f(loc_color, R, G, B, 1.0)

    glDrawArrays(GL_TRIANGLES, offset, num_triangulos * 3)


# Função que desenha uma estrela 2D inclinada
def desenha_estrela(offset, num_triangulos, tx, ty, tz, color, angulo_z=0.0):
    rad_z = np.radians(angulo_z)
    cos_z = np.cos(rad_z)
    sin_z = np.sin(rad_z)

    # Rotação em Z (para girar a estrela)
    mat_rot_z = np.array([
        cos_z, -sin_z, 0.0, 0.0,
        sin_z,  cos_z, 0.0, 0.0,
        0.0,    0.0,   1.0, 0.0,
        0.0,    0.0,   0.0, 1.0
    ], dtype=np.float32)

    # Translação
    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    mat_transform = multiplica_matriz(mat_transl, mat_rot_z)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)

    R, G, B = color
    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLES, offset, num_triangulos * 3)

# Função que desenha uma flor 2D
def desenha_flor(offset, num_petalas, tx, ty, tz, cor_petala, cor_miolo, s_i):
    # Matriz de transformação
    mat_trans = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    mat_scale = np.array([
        s_i,  0.0, 0.0, 0.0, 
        0.0,  s_i, 0.0, 0.0, 
        0.0,  0.0, s_i, 0.0, 
        0.0,  0.0, 0.0, 1.0
    ], np.float32)

    mat_transform = multiplica_matriz(mat_trans, mat_scale)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)      

    # Desenha pétalas 
    R, G, B = cor_petala
    glUniform4f(loc_color, R, G, B, 1.0)
    for i in range(num_petalas):
        glDrawArrays(GL_TRIANGLES, offset + i * 3, 3)

    # Desenha miolo 
    R, G, B = cor_miolo
    glUniform4f(loc_color, R, G, B, 1.0)
    for i in range(num_petalas):
        base = offset + num_petalas * 3 + i * 3
        glDrawArrays(GL_TRIANGLES, base, 3)

# Função que desenha a chama (piramide) da cena
def desenhar_chama(offset, tx, ty, tz, angulo_x, angulo_y):

    rad_x = np.radians(angulo_x)
    rad_y = np.radians(angulo_y)

    cos_x = np.cos(rad_x)
    sin_x = np.sin(rad_x)
    cos_y = np.cos(rad_y)
    sin_y = np.sin(rad_y)

    # Matriz de Translação
    mat_transl = np.array([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)

    # Matriz de Rotação X
    mat_rot_x = np.array([
        1.0, 0.0,    0.0,   0.0,
        0.0, cos_x, -sin_x, 0.0,
        0.0, sin_x,  cos_x, 0.0,
        0.0, 0.0,    0.0,   1.0
    ], dtype=np.float32)

    # Matriz de Rotação Y
    mat_rot_y = np.array([
        cos_y, 0.0, sin_y, 0.0,
        0.0,   1.0, 0.0,   0.0,
       -sin_y, 0.0, cos_y, 0.0,
        0.0,   0.0, 0.0,   1.0
    ], dtype=np.float32)

    # Matriz final de transformação
    mat_transform = multiplica_matriz(mat_rot_y, mat_rot_x)
    mat_transform = multiplica_matriz(mat_transl, mat_transform)

    loc_mat = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc_mat, 1, GL_TRUE, mat_transform)  

    # Cores para simular o fogo
    cores = [
        (1.0, 0.2, 0.0),  
        (1.0, 0.8, 0.0),  
        (0.9, 0.0, 0.0), 
        (1.0, 0.5, 0.0),  
        (0.7, 0.0, 0.0),  
    ]

    # Define as cores das 4 faces triangulares e a base
    for i in range(5):
        R, G, B = cores[i]
        glUniform4f(loc_color, R, G, B, 1.0)
        glDrawArrays(GL_TRIANGLES, offset + i * 3, 3)
 
 # incrementos para translacao
x_inc = 0.0
y_inc = 0.0

# incrementos para rotacao
r_inc = 0.0

# coeficiente de escala
s_inc = 0.5

# Função para ler ações do teclado
def key_event(window,key,scancode,action,mods):
    global x_inc, y_inc, r_inc, s_inc
    
    # Translação do Sol
    if x_inc > -4.5 and x_inc <= 0.1:
        if key == 263: x_inc -= 0.025 # esquerda
        if key == 262: x_inc += 0.025 # direita
    else:
        x_inc = 0

    # Rotação da Estrelas
    if key == 265: r_inc += 2 # cima (rotação horaria)
    if key == 264: r_inc -= 2 # baixo (rotação anti - horaria)

    # Escala das Flores
    if s_inc < 3.5 and s_inc > 0 :
        if key == 90: s_inc += 0.1 #letra z
        if key == 88: s_inc -= 0.1 #letra x
    else:
        s_inc = 0.5

    # Combinação das 3 operações
    if key == 84: #letra t
        s_inc += 0.03 if s_inc < 3.5 and s_inc > 0 else s_inc == 0.5
        x_inc -= 0.025 if x_inc > -4.5 and x_inc <= 0.1 else x_inc == 0
        r_inc += 2 
    
    # Malha Poligonal
    if key == 80:glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)   # letra p
    if key == 70: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) # letra f

glfw.set_key_callback(window,key_event)

# -------------- Programa Central --------------------------
glEnable(GL_DEPTH_TEST)

# Configuração inicial, com a criação dos vertices
vertices_sol = criar_esfera(0.3)
vertices_topo_arvore = criar_esfera(0.28)
vertices_pao = criar_pao(0.07, 0.07, 0.25)
vertices_ret = criar_retangulo(2)
vertices_lona = criar_lona(0.8, 0.8)
vertices_cilindro = criar_cilindro()
vertices_lua = criar_esfera(0.3)
vertices_estrela_1 = criar_estrela(6, 0.1, 0.05)
vertices_estrela_2 = criar_estrela(5, 0.1, 0.05)
vertices_maca = criar_esfera(0.06, 12, 12)
vertices_cabo = criar_cilindro(0.01, 15, 15, 0.01)
vertices_flor = criar_flor(8, 0.01, 0.02)
vertices_chama = criar_piramide(0.03)
vertices_vela = criar_cilindro(0.03, 20, 20, 0.18)
vertices_vela2 = criar_cilindro(0.03, 20, 20, 0.005)

todos_vertices = np.concatenate((vertices_sol, 
                                 vertices_topo_arvore,
                                 vertices_ret, 
                                 vertices_lona,
                                 vertices_cilindro,
                                 vertices_pao,
                                 vertices_lua,
                                 vertices_estrela_1,
                                 vertices_estrela_2,
                                 vertices_maca,
                                 vertices_cabo,
                                 vertices_flor,
                                 vertices_chama,
                                 vertices_vela,
                                 vertices_vela2
                                 ), dtype=np.float32)

# Calcula os offsets para cada objeto
offset_sol = 0
offset_topo_arvore = offset_sol + len(vertices_sol)
offset_ret = offset_topo_arvore + len(vertices_topo_arvore) 
offset_lona = offset_ret + len(vertices_ret)
offset_cilindro = offset_lona + len(vertices_lona)
offset_pao = offset_cilindro + len(vertices_cilindro)
offset_lua = offset_pao+ len(vertices_pao)
offset_estrela1 = offset_lua + len(vertices_lua)
offset_estrela2 = offset_estrela1 + len(vertices_estrela_1)
offset_maca = offset_estrela2 + len(vertices_estrela_2)
offset_cabo = offset_maca + len(vertices_maca)
offset_flor = offset_cabo + len(vertices_cabo)
offset_chama = offset_flor + len(vertices_flor)
offset_vela = offset_chama + len(vertices_chama)
offset_vela2 = offset_vela + len(vertices_vela)

# Configura o VBO uma única vez
setup_VBO(todos_vertices)

# Loop principal
glfw.show_window(window)
while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    # Sol
    desenha_esfera(offset_sol, len(vertices_sol)//3, 0.85+x_inc, 0.88, 1, (1.0, 0.8, 0.2), (1.0, 0.5, 0.1))
    desenha_esfera(offset_lua, len(vertices_lua)//3, 3.5+x_inc, 0.88, -0.3, (0.75, 0.75, 0.8), (0.8, 0.9, 1.0))

    # Arvore
    desenha_esfera(offset_topo_arvore, len(vertices_topo_arvore)//3, -0.6, 0.29, -0.7, (0.4, 0.7, 0.3), (0.2, 0.5, 0.2))
    desenha_cilindro(offset_cilindro, len(vertices_cilindro)//3, -0.6, -0.4, 0.1, (0.36, 0.25, 0.2))

    # Fundo da cena
    x = 3.5 + x_inc
    color_background = (0.05, 0.05, 0.2) if x < 1 else (0.4, 1, 0.8)
    color_grass = (0.08, 0.45, 0.2) if x < 1 else (0.2, 0.8, 0.3)
    desenha_quadrilatero(offset_ret, len(vertices_ret)//3, 0.0, 0.8, 0.99, -90.0, color_background)
    desenha_quadrilatero(offset_ret, len(vertices_ret)//3, 0.0, 1, 0.99, -90.0, color_background)

    desenha_quadrilatero(offset_ret, len(vertices_ret)//3, 0.0, -1.2, 0.99, 90.0, color_grass)

    # Lona
    desenha_quadrilatero(offset_lona, len(vertices_lona)//3, 0, -0.8, 0.8, 35, (1, 0.3, 0.3))

    # Pão
    desenha_pao(offset_pao, 0.4, -0.5, -0.1, -18, 20)
    
    # Estrelas
    if x < 1 :
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, 0.0, 0.6, -0.1, (1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, -0.5, 0.8, -0.1, (1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, 0.2, 0.5, -0.1, (1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, -0.7, 0.9, -0.1, (1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, 0.7, 0.5, -0.1, (1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, 0.4, 0.7, -0.1, (1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, -0.2, 0.4, -0.1, (1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, 0.4, 0.2, -0.1, (1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, 0.1, 0.9, -0.1, (1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_2)//3, -0.6, 0.3, -0.1, (1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, -0.3, 0.85, -0.1,(1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, -0.1, 0.1, -0.1,(1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, 0.2, 0, -0.1,(1.0, 0.75, 0.2), 45.0+r_inc)
        desenha_estrela(offset_estrela1, len(vertices_estrela_1)//3, 0.8, 0.3, -0.1,(1.0, 0.9, 0.3), 45.0+r_inc)
        desenha_estrela(offset_estrela2, len(vertices_estrela_2)//3, 0.9, 0.7, -0.1,(1.0, 0.75, 0.2), 45.0+r_inc)

    # Limão
    desenha_esfera(offset_maca, len(vertices_maca)//3, 0.15, -0.55, -0.1, (0.8, 1.0, 0.2), (0.7, 1.0, 0.3))
    desenha_cilindro(offset_cabo,len(vertices_cabo)//3, 0.15, -0.5, -0.1, (0, 1, 0.1))

    # Laranja
    desenha_esfera(offset_maca, len(vertices_maca)//3, 0, -0.6, -0.1, (1.0, 0.5, 0.0), (0.9, 0.55, 0.3))
    desenha_cilindro(offset_cabo,len(vertices_cabo)//3, 0, -0.545, -0.1, (0, 1, 0.1))

    # Maça
    desenha_esfera(offset_maca, len(vertices_maca)//3, 0.08, -0.45, -0.1,(1.0, 0, 0.2), (1, 0.2, 0))
    desenha_cilindro(offset_cabo,len(vertices_cabo)//3, 0.08, -0.395, -0.1, (0, 1, 0.1))

    #Flores
    desenha_flor(offset_flor, 8, -0.8, -0.3, -0.1, (1.0, 1.0, 1.0), (1.0, 0.95, 0.5), s_inc)
    desenha_flor(offset_flor, 8, -0.65, -0.5, -0.1, (0.98, 0.98, 0.96), (1.0, 0.85, 0.4), s_inc)
    desenha_flor(offset_flor, 8, -0.85, -0.52, -0.1, (0.99, 0.99, 0.98), (1.0, 0.9, 0.45), s_inc)
    desenha_flor(offset_flor, 8, -0.45, -0.65, -0.1, (1.0, 1.0, 0.99), (1.0, 0.8, 0.35), s_inc)
    desenha_flor(offset_flor, 8, -0.35, -0.55, -0.1, (0.97, 0.97, 0.95), (1.0, 0.88, 0.42), s_inc)
    desenha_flor(offset_flor, 8, -0.05, -0.25, -0.1, (1.0, 1.0, 1.0), (1.0, 0.92, 0.48), s_inc)
    desenha_flor(offset_flor, 8, -0.4, -0.28, -0.1, (0.98, 0.98, 0.97), (1.0, 0.87, 0.38), s_inc)
    desenha_flor(offset_flor, 8, -0.17, -0.35, -0.1, (0.99, 0.99, 0.99), (1.0, 0.82, 0.32), s_inc)
    desenha_flor(offset_flor, 8, -0.7, -0.7 , -0.1, (1.0, 1.0, 0.98), (1.0, 0.93, 0.47), s_inc)
    desenha_flor(offset_flor, 8, -0.8, -0.8, -0.1, (0.98, 0.98, 0.94), (1.0, 0.83, 0.37), s_inc)
    desenha_flor(offset_flor, 8, -0.5, -0.9, -0.1, (0.99, 0.99, 0.97), (1.0, 0.89, 0.43), s_inc)
    desenha_flor(offset_flor, 8, 0.7, -0.9, -0.1, (1.0, 1.0, 1.0), (1.0, 0.84, 0.39), s_inc)
    desenha_flor(offset_flor, 8, 0.65, -0.75, -0.1, (0.97, 0.97, 0.96), (1.0, 0.94, 0.49), s_inc)
    desenha_flor(offset_flor, 8, 0.9, -0.65, -0.1, (0.98, 0.98, 0.98), (1.0, 0.81, 0.33), s_inc)

    # Vela
    desenhar_chama(offset_chama, 0.7, -0.25, -0.5, -18 , 20)
    desenha_cilindro(offset_vela,len(vertices_vela)//3, 0.7, -0.4, -0.2,(0.7, 0.65, 0.6))
    desenha_cilindro(offset_vela2,len(vertices_vela2)//3, 0.7, -0.25, -0.3, (1.0, 0.95, 0.6))
    
    glfw.swap_buffers(window)
    glfw.poll_events()
    
glfw.terminate()
