import numpy as np
import cv2
import streamlit as st

# Layout
logo = 'favicon.ico'
st.set_page_config(page_title='Detecção de Distanciamento Social',
                   page_icon=logo, layout='wide',
                   initial_sidebar_state='expanded'
                   )

st.subheader('⚠️Monitoramento de Distância Social⚠️', divider='rainbow')

# carregando arquivos de configuração

configFile = 'MobileNetSSD_deploy.prototxt'
modelFile = 'MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


### Funções
def detect(frame, network):
    """
    Detecta humanos em um quadro de imagem e retorna suas caixas delimitadoras e centros.

    Parâmetros:
    frame (numpy.array): Uma imagem em formato de array numpy.
    network (cv2.dnn_Net): Uma rede neural pré-carregada.

    Retorno:
    results (list): Uma lista de tuplas. Cada tupla contém a confiança da detecção,
                    a caixa delimitadora e o centro do objeto detectado.
    """

    # Lista que armazenara os resultados da detecçcão
    results = []

    # Obtém as dimensões da imagem
    h, w = frame.shape[:2]

    # Pré-processamento: subtração da média e redimensionamento para coincidir com o conjunto de treinamento do modelo.
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), [127.5, 127.5, 127.5])

    # Define o blob como entrada para a rede
    network.setInput(blob)

    # Realiza uma inferência do modelo, passando o blob pela rede.
    network_output = network.forward()

    # Percorre todos os resultados.
    for i in np.arange(0, network_output.shape[2]):

        # Obtém a classe e a confiança da detecção
        class_id = network_output[0, 0, i, 1]
        confidence = network_output[0, 0, i, 2]

        # Filtra para pessoas detectadas (classID 15) e com alta confiança.
        # https://github.com/chuanqi305/MobileNet-SSD/blob/master/demo.py#L21
        if confidence > 0.7 and class_id == 15:
            # Remapeia as saídas de posição 0-1 para o tamanho da imagem para a caixa delimitadora.
            box = network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')

            # Calcula o centro da pessoa a partir da caixa delimitadora.
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            # Adiciona os resultados à lista
            results.append((confidence, box, (center_x, center_y)))

    return results

def detect_violations(results):
    """
    Detecta se existem pessoas que estão muito próximas uma das outras de forma insegura.

    Parâmetros:
    results (list): Lista de resultados contendo informações sobre as detecções. Cada detecção é representada
    por uma tupla contendo a classe da detecção, as coordenadas do retângulo delimitador e as coordenadas do centróide.

    Retorna:
    violations (set): Conjunto contendo os índices das detecções que estão muito próximas uma das outras.
    """

    # Inicializa o conjunto de violações
    violations = set()

    # Multiplicador na largura em pixel da menor detecção.
    fac = 1.2

    # Se existirem 2 ou mais detecções
    if len(results) >= 2:
        # A largura é a borda direita menos a esquerda.
        boxes_width = np.array([abs(int(r[1][2] - r[1][0])) for r in results])

        # Obtém os centróides de todas as detecções
        centroids = np.array([r[2] for r in results])

        # Calcula a matriz de distâncias entre todos os pares de centróides
        distance_matrix = euclidean_dist(centroids, centroids)

        # Para cada detecção inicial...
        for row in range(distance_matrix.shape[0]):
            # Compara a distância com todas as outras detecções restantes.
            for col in range(row + 1, distance_matrix.shape[1]):
                # Presume insegurança se estiver mais perto do que 1.2x (fac) a largura de uma pessoa.
                ref_distance = int(fac * min(boxes_width[row], boxes_width[col]))

                # Se a distância for menor que a distância de referência, adiciona ao conjunto de violações
                if distance_matrix[row, col] < ref_distance:
                    violations.add(row)
                    violations.add(col)
    # Retorna o conjunto de violações
    return violations

def euclidean_dist(A, B):
    """
    Calcula a distância euclidiana par a par entre cada combinação de centróides.

    Parâmetros:
    centroides_A (numpy.array): Array de centróides. Cada centróide é representado por um vetor de coordenadas.
    centroides_B (numpy.array): Outro array de centróides.

    Retorna:
    matriz_distancias (numpy.array): Matriz de distâncias de tamanho len(centroides_A) por len(centroides_B).
    """

    # Calcula o quadrado da norma de cada vetor em A e expande a última dimensão
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]

    # Calcula o quadrado da norma de cada vetor em B
    p2 = np.sum(B**2, axis=1)

    # Calcula o produto interno entre cada vetor em A e em B, e multiplica por -2
    p3 = -2 * np.dot(A, B.T)

    # Calcula a raiz quadrada da soma de p1, p2 e p3, resultando na distância euclidiana, e arredonda para 2 casas decimais
    matriz_distancias = np.round(np.sqrt(p1 + p2 + p3), 2)

    return matriz_distancias

###

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name
###

# Sidebar
st.sidebar.header('Detecção de Distanciamento Social')
st.sidebar.markdown('''
   ## **Detalhes do Projeto:**
- **Objetivo:** Utilizar visão computacional para detectar violações de distanciamento social em um ambiente.
- **Ferramentas e Tecnologias:** Python, OpenCV, Streamlit
- **Modelo de Detecção:** MobileNet SSD

---
''')

image_path = "logo.png"
text = "Developed and Maintained by: Rian.Bispo"
rodape = st.sidebar.image(image_path, caption=text)

st.sidebar.markdown('''  
  [![Linkedin Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/rian-bispo/)](https://www.linkedin.com/in/rian-bispo/)
''')
###

# Define se o vídeo deve ser exibido durante o processamento
SHOW_VIDEO = 0

# Caminho do vídeo de entrada
INPUT_PATH = st.file_uploader("Escolha um Video", type='mp4')

# Caminho do vídeo de saída
OUTPUT_PATH = 'output.mp4'

# Inicia a captura de vídeo
if INPUT_PATH is not None:
    # Converta o objeto retornado por st.file_uploader para um caminho de arquivo
    video_path = save_uploaded_file(INPUT_PATH)

    cap = cv2.VideoCapture(video_path)

    # Inicializa o escritor de vídeo
    writer = None

    # Inicializa o tempo do frame anterior
    prev_frame_time = 0

    # Inicializa o tempo do novo frame
    new_frame_time = 0

    # Inicializa o contador
    counter = 0

    print("Processando os frames, por favor aguarde ...")
    st.warning("Processando os frames, por favor aguarde ...")

    # Enquanto houver frames para processar
    while cap.isOpened():
        # Lê o próximo frame
        ret, frame = cap.read()

        # Se não houver mais frames, encerra o loop
        if not ret:
            break

        # Detecta as caixas delimitadoras dos objetos no frame atual
        results = detect(frame, network=net)

        # Detecta quais caixas delimitadoras estão muito próximas (i.e. as violações)
        violations = detect_violations(results)

        t, _ = net.getPerfProfile()
        label = 'Tempo de inferência: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

        # Desenha todas as caixas delimitadoras e indica se estão em violação
        for index, (prob, bounding_box, centroid) in enumerate(results):
            start_x, start_y, end_x, end_y = bounding_box

            # Se a caixa delimitadora estiver em violação, a cor é vermelha. Senão, a cor é verde.
            color = (0, 0, 255) if index in violations else (0, 255, 0)

            # Desenha a caixa delimitadora
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            # Escreve o tempo de inferência no frame
            cv2.putText(
                frame, label,
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))

            # Escreve se a caixa delimitadora está em violação
            cv2.putText(
                frame, 'Inseguro' if index in violations else 'Seguro',
                (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Escreve o número total de violações
            cv2.putText(
                frame, f'Num Violations: {len(violations)}',
                (10, frame.shape[0] - 25),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.0, color=(0, 0, 255), thickness=1)

        # Se SHOW_VIDEO for verdadeiro, exibe o frame
        if SHOW_VIDEO:
            cv2.imshow('frame', frame)
            # Se a tecla 'q' for pressionada, encerra o loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Se o escritor de vídeo ainda não foi inicializado
        if writer is None:
            # Define o codec de vídeo
            # Para stremlit
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            #Para execução local
            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Inicializa o escritor de vídeo
            writer = cv2.VideoWriter(
                OUTPUT_PATH, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # Se o escritor de vídeo foi inicializado, escreve o frame no arquivo de saída
        if writer:
            writer.write(frame)

    # Libera o capturador de vídeo
    cap.release()

    # Libera o escritor de vídeo
    writer.release()

    print(f'Finalizado. O vídeo foi salvo em {OUTPUT_PATH}')
    st.success(f'Finalizado. O vídeo foi salvo em {OUTPUT_PATH}')

    # Exibir o vídeo
    video_file = open(OUTPUT_PATH, "rb").read()
    st.video(video_file)

    # Botão de download
    with open(OUTPUT_PATH, "rb") as file:
        btn = st.download_button(
            label="Download Video Resultante",
            data=file,
            file_name="output.mp4",
            mime="video/mp4"
        )

    # Fecha todas as janelas
    cv2.destroyAllWindows()

    print('Todas as janelas foram fechadas...')
