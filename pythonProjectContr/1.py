import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import logging
import subprocess
import sys
import json
from web3 import Web3
import csv

# Функция для установки пакета
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Проверка установки пакета kaleido
try:
    import kaleido
except ImportError:
    install('kaleido')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("info.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# URL узла Ethereum (например, Infura или локальный узел)
eth_node_url = "http://127.0.0.1:7545"

try:
    # Подключение к сети Ethereum
    web3 = Web3(Web3.HTTPProvider(eth_node_url))

    if web3.is_connected():
        logger.info("Successfully connected to Ethereum network")
        block_number = web3.eth.get_block('latest')
        print(f"Current block number: {block_number}")
    else:
        logger.error("Failed to connect to Ethereum network")

except Exception as e:
    logger.error(f"Exception occurred: {e}")
    print(f"Exception occurred: {e}")

# Путь к конфигурационному файлу
config_path = 'config.json'
if not os.path.exists(config_path):
    logger.error(f"Файл конфигурации {config_path} не найден.")
    exit()

# Загрузка конфигурации
with open(config_path) as config_file:
    config = json.load(config_file)

# Конфигурационные параметры
file_path = config["file_path"]
threshold = config["threshold"]

# Проверка подключения
if not web3.is_connected():
    logger.error("Failed to connect to Ethereum network")
    exit()

# ABI и адрес контракта
abi_path = 'build/contracts/MyContract.json'
if not os.path.exists(abi_path):
    logger.error(f"Файл {abi_path} не найден.")
    exit()

with open(abi_path) as f:
    contract_info = json.load(f)
abi = contract_info['abi']
contract_address = contract_info['networks']['5777']['address']
contract = web3.eth.contract(address=contract_address, abi=abi)

def store_data_in_blockchain(id, data):
    try:
        tx_hash = contract.functions.storeData(id, data).transact({
            'from': web3.eth.accounts[0],
            'gas': 2000000
        })
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        logger.error(f"Ошибка записи данных в блокчейн: {e}")
        raise RuntimeError(f"Ошибка записи данных в блокчейн: {e}")

def get_data_from_blockchain(index):
    try:
        return contract.functions.getData(index).call()
    except Exception as e:
        logger.error(f"Ошибка получения данных из блокчейна: {e}")
        raise RuntimeError(f"Ошибка получения данных из блокчейна: {e}")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='cp1251', delimiter=';')
        logger.info("Данные успешно загружены из %s", file_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных из файла {file_path}: {e}")
        raise ValueError(f"Ошибка загрузки данных из файла {file_path}: {e}")
    return data

def clean_data(data):
    if data.isnull().values.any():
        data = data.dropna()
        logger.info("Пропущенные значения удалены.")
    return data

def classify_threats(data):
    if 'Вероятность угрозы (%)' not in data.columns or 'Финансовые потери за инцидент (USD)' not in data.columns:
        logger.error("Отсутствуют необходимые столбцы.")
        raise ValueError(
            "Отсутствуют необходимые столбцы: 'Вероятность угрозы (%)' или 'Финансовые потери за инцидент (USD)'")
    data['Оценка риска'] = (data['Вероятность угрозы (%)'] / 100) * data['Финансовые потери за инцидент (USD)']
    logger.info("Оценка риска завершена.")
    return data

def build_risk_model(data):
    if 'Оценка риска' not in data.columns:
        logger.error("Отсутствует столбец 'Оценка риска'.")
        raise ValueError("Отсутствует столбец 'Оценка риска'")
    X = np.log1p(data[['Вероятность угрозы (%)', 'Финансовые потери за инцидент (USD)']])
    y = np.log1p(data['Оценка риска'])
    model = LinearRegression().fit(X, y)
    logger.info("Модель риска построена.")
    return model

def visualize_risks(data):
    try:
        logger.info("Создание scatter plot...")
        fig = px.scatter(data, x='Вероятность угрозы (%)', y='Финансовые потери за инцидент (USD)',
                         color='Оценка риска', title='Анализ рисков', labels={'Оценка риска': 'Уровень риска'})
        scatter_path = 'risk_analysis_scatter.png'
        fig.write_image(scatter_path)
        fig.show()

        if data['Вероятность угрозы (%)'].nunique() < 4 or data['Финансовые потери за инцидент (USD)'].nunique() < 4:
            logger.warning("Недостаточно уникальных значений для создания категорий.")
            return scatter_path, None

        logger.info("Создание heatmap...")
        data['Категория вероятности'] = pd.cut(data['Вероятность угрозы (%)'], bins=4,
                                               labels=['Низкая', 'Средняя', 'Высокая', 'Очень высокая'])
        data['Категория потерь'] = pd.cut(data['Финансовые потери за инцидент (USD)'], bins=4,
                                          labels=['Низкие', 'Средние', 'Высокие', 'Очень высокие'])

        heatmap_data = pd.pivot_table(data, values='Оценка риска', index='Категория вероятности',
                                      columns='Категория потерь', aggfunc=np.mean, observed=True)

        if heatmap_data.empty:
            logger.warning("Нет данных для тепловой карты.")
            return scatter_path, None

        fig = px.imshow(heatmap_data, title='Тепловая карта рисков', labels={'color': 'Уровень риска'})
        heatmap_path = 'risk_analysis_heatmap.png'
        fig.write_image(heatmap_path)
        fig.show()

        logger.info("Визуализация создана.")
        return scatter_path, heatmap_path
    except Exception as e:
        logger.error(f"Ошибка визуализации данных: {e}")
        raise RuntimeError(f"Ошибка визуализации данных: {e}")

def evaluate_model(model, data):
    try:
        logger.info("Оценка модели...")
        residuals = model.predict(
            np.log1p(data[['Вероятность угрозы (%)', 'Финансовые потери за инцидент (USD)']])) - np.log1p(
            data['Оценка риска'])
        plt.figure()
        plt.hist(residuals, bins=20)
        plt.title('Гистограмма остатков')
        residuals_path = 'residuals_histogram.png'
        plt.savefig(residuals_path)
        plt.show()
        logger.info("Оценка модели завершена.")
        return residuals_path
    except Exception as e:
        logger.error(f"Ошибка оценки модели: {e}")
        raise RuntimeError(f"Ошибка оценки модели: {e}")

def generate_recommendations(data):
    if 'Оценка риска' not in data.columns:
        logger.error("Отсутствует столбец 'Оценка риска'.")
        raise ValueError("Отсутствует столбец 'Оценка риска'")
    high_risk = data[data['Оценка риска'] > threshold]
    logger.info("Рекомендации по высокому уровню риска сгенерированы.")
    return high_risk

def create_report(data, visuals_paths):
    try:
        pdf = FPDF()
        pdf.add_page()
        font_path = 'DejaVuSansCondensed.ttf'
        if not os.path.exists(font_path):
            logger.error(f"TTF Font file not found: {font_path}")
            raise RuntimeError(f"TTF Font file not found: {font_path}")
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        pdf.cell(200, 10, txt="Отчет по анализу рисков", ln=True, align='C')
        pdf.cell(200, 10, txt="Риски с высоким уровнем:", ln=True)
        for index, row in data.iterrows():
            pdf.cell(200, 10, txt=f"{row['Название угрозы']}: {row['Оценка риска']}", ln=True)

        for path in visuals_paths:
            if path:
                pdf.add_page()
                pdf.image(path, x=10, y=10, w=190)

        pdf.output("Отчет_по_анализу_рисков.pdf")
        logger.info("Отчет успешно создан.")
    except Exception as e:
        logger.error(f"Ошибка создания отчета: {e}")
        raise RuntimeError(f"Ошибка создания отчета: {e}")

    if __name__ == "__main__":
        try:
            data = load_data(file_path)
            data = clean_data(data)
            data = classify_threats(data)
            model = build_risk_model(data)
            scatter_path, heatmap_path = visualize_risks(data)
            residuals_path = evaluate_model(model, data)
            high_risk = generate_recommendations(data)
            create_report(high_risk, [scatter_path, heatmap_path, residuals_path])

            # Блокчейн интеграция
            with open('data.csv', mode='r', encoding='cp1251') as file:
                reader = csv.reader(file, delimiter=';')
                for index, row in enumerate(reader):
                    logger.info(f"Processing row {index}: {row}")
                    if len(row) > 1:  # Учитываем все значения
                        # Можно использовать подходящий индекс для записи данных в блокчейн
                        id = index
                        data_str = json.dumps(row)  # Преобразуем всю строку в строку JSON
                        receipt = store_data_in_blockchain(id, data_str)
                        logger.info(f'Data stored in transaction: {receipt.transactionHash.hex()}')
                    else:
                        logger.error(f"Неверный формат данных в строке {index}: {row}")

            total_records = contract.functions.getRecordsCount().call()
            for i in range(total_records):
                id, data, timestamp = get_data_from_blockchain(i)
                logger.info(f'Record {i}: ID={id}, Data={data}, Timestamp={timestamp}')
        except Exception as e:
            logger.error(f"Ошибка выполнения программы: {e}")
            raise RuntimeError(f"Ошибка выполнения программы: {e}")

