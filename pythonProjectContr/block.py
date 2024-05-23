from web3 import Web3
import csv
import json

# Подключение к локальной Ethereum сети (Ganache)
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Проверка подключения
if not web3.isConnected():
    print("Failed to connect to the blockchain.")
    exit()

# Учетная запись для отправки транзакций
account = web3.eth.accounts[0]

# ABI и адрес контракта
with open('build/contracts/MyContract.json') as f:
    contract_info = json.load(f)
abi = contract_info['abi']
contract_address = contract_info['networks']['5777']['address']
contract = web3.eth.contract(address=contract_address, abi=abi)

# Функция для записи данных в блокчейн
def store_data(id, data):
    tx_hash = contract.functions.storeData(id, data).transact({
        'from': account,
        'gas': 2000000
    })
    receipt = web3.eth.waitForTransactionReceipt(tx_hash)
    return receipt

# Чтение данных из CSV и запись в блокчейн
with open('data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        id, data = row
        receipt = store_data(int(id), data)
        print(f'Data stored in transaction: {receipt.transactionHash.hex()}')

# Функция для получения данных из блокчейна
def get_data(index):
    return contract.functions.getData(index).call()

# Получение всех записей из блокчейна
total_records = contract.functions.getRecordsCount().call()
for i in range(total_records):
    id, data, timestamp = get_data(i)
    print(f'Record {i}: ID={id}, Data={data}, Timestamp={timestamp}')
