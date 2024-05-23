// MyContract.sol
pragma solidity ^0.8.0;

contract MyContract {
    struct DataRecord {
        uint id;
        string data;
        uint timestamp;
    }

    DataRecord[] public dataRecords;

    function storeData(uint _id, string memory _data) public {
        dataRecords.push(DataRecord(_id, _data, block.timestamp));
    }

    function getData(uint _index) public view returns (uint, string memory, uint) {
        DataRecord memory record = dataRecords[_index];
        return (record.id, record.data, record.timestamp);
    }

    function getRecordsCount() public view returns (uint) {
        return dataRecords.length;
    }
}
