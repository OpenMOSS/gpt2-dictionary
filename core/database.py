import os
from typing import Dict

from bson import ObjectId
import numpy as np
import pymongo
import gridfs
import pymongo.database

from core.utils.bytes import bytes_to_np, np_to_bytes

class MongoClient:
    def __init__(self, mongo_uri: str, mongo_db: str):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[mongo_db]
        self.fs = gridfs.GridFS(self.db)
        self.feature_collection = self.db['features']
        self.dictionary_collection = self.db['dictionaries']
        self.attn_head_collection = self.db['attn_heads']
        self.dictionary_collection.create_index([('name', pymongo.ASCENDING), ('series', pymongo.ASCENDING)], unique=True)
        self.feature_collection.create_index([('dictionary_id', pymongo.ASCENDING), ('index', pymongo.ASCENDING)], unique=True)
        self.attn_head_collection.create_index([('dictionary_id', pymongo.ASCENDING), ('index', pymongo.ASCENDING)], unique=True)

    def _to_gridfs(self, data):
        """
        Recursively convert numpy arrays in data object to bytes, and store in GridFS
        """
        if isinstance(data, dict):
            return {k: self._to_gridfs(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_gridfs(v) for v in data]
        if isinstance(data, np.ndarray):
            return self.fs.put(np_to_bytes(data))
        return data
    
    def _from_gridfs(self, data):
        """
        Recursively convert GridFS references in data object to numpy arrays
        """
        if isinstance(data, dict):
            return {k: self._from_gridfs(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._from_gridfs(v) for v in data]
        if isinstance(data, ObjectId) and self.fs.exists(data):
            return bytes_to_np(self.fs.get(data).read())
        return data

    def create_dictionary(self, dictionary_name: str, n_features: int, dictionary_series: str | None = None):
        dict_id = self.dictionary_collection.insert_one({'name': dictionary_name, 'n_features': n_features, 'series': dictionary_series}).inserted_id
        self.feature_collection.insert_many([
            {
                'dictionary_id': dict_id,
                'index': i
            }
            for i in range(n_features)
        ])

    def update_feature(self, dictionary_name: str, feature_index: int, feature_data: Dict, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        assert dictionary is not None, f'Dictionary {dictionary_name} not found'
        feature = self.feature_collection.find_one({'dictionary_id': dictionary['_id'], 'index': feature_index})
        assert feature is not None, f'Feature {feature_index} not found in dictionary {dictionary_name}'
        self.feature_collection.update_one({'_id': feature['_id']}, {'$set': self._to_gridfs(feature_data)})

    def list_dictionaries(self, dictionary_series: str | None = None):
        return [d['name'] for d in self.dictionary_collection.find({'series': dictionary_series} if dictionary_series is not None else {})]
    
    def get_feature(self, dictionary_name: str, feature_index: int, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        if dictionary is None:
            return None
        feature = self.feature_collection.find_one({'dictionary_id': dictionary['_id'], 'index': feature_index})
        if feature is None:
            return None
        return self._from_gridfs(feature)

    def get_random_alive_feature(self, dictionary_name: str, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        if dictionary is None:
            return None
        feature = self.feature_collection.aggregate([
            {'$match': {'dictionary_id': dictionary['_id'], 'max_feature_acts': {'$gt': 0}}},
            {'$sample': {'size': 1}}
        ]).next()
        if feature is None:
            return None
        return self._from_gridfs(feature)
    
    def get_alive_feature_count(self, dictionary_name: str, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        if dictionary is None:
            return None
        return self.feature_collection.count_documents({'dictionary_id': dictionary['_id'], 'max_feature_acts': {'$gt': 0}})
    
    def get_max_feature_acts(self, dictionary_name: str, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        if dictionary is None:
            return None
        pipeline = [
            {'$match': {'dictionary_id': dictionary['_id'], 'max_feature_acts': {'$gt': 0}}},
            {'$project': {'_id': 0, 'index': 1, 'max_feature_acts': 1}}
        ]
        return {f['index']: f['max_feature_acts'] for f in self.feature_collection.aggregate(pipeline)}
    
    def get_feature_act_times(self, dictionary_name: str, dictionary_series: str | None = None):
        dictionary = self.dictionary_collection.find_one({'name': dictionary_name, 'series': dictionary_series})
        if dictionary is None:
            return None
        pipeline = [
            {'$match': {'dictionary_id': dictionary['_id'], 'max_feature_acts': {'$gt': 0}}},
            {'$project': {'_id': 0, 'index': 1, 'act_times': 1}}
        ]
        return {f['index']: f['act_times'] for f in self.feature_collection.aggregate(pipeline)}

    
    def get_attn_head(self, dictionary_name: str, head_index: int, dictionary_series: str | None = None):
        pipeline = [
            {
                "$lookup": {
                    "localField": "dictionary_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "dictionary",
                }
            },
            {"$unwind": "$dictionary"},
            {
                "$match": {
                    "dictionary.name": dictionary_name,
                    "index": head_index,
                    "dictionary.series": dictionary_series,
                }
            },
            {
                "$project": {
                    "dictionary_id": 0,
                }
            }
        ]
        attn_head = self.attn_head_collection.aggregate(pipeline).next()
        if len(attn_head["attn_score"]) == 0:
            return attn_head
        pipeline = [
            *pipeline,
            {"$unwind": "$attn_scores"},
            {
                "$lookup": {
                    "localField": "attn_scores.dictionary1_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "attn_scores.dictionary1",
                }
            },
            {"$unwind": "$attn_scores.dictionary1"},
            {
                "$lookup": {
                    "localField": "attn_scores.dictionary2_id",
                    "from": "dictionaries",
                    "foreignField": "_id",
                    "as": "attn_scores.dictionary2",
                }
            },
            {"$unwind": "$attn_scores.dictionary2"},
            {
                "$project": {
                    "attn_scores.dictionary1_id": 0,
                    "attn_scores.dictionary2_id": 0,
                }
            },
            {
                "$group": {
                    "_id": "$_id",
                    "attn_scores": {"$push": "$attn_scores"},
                    "dictionary": {"$first": "$dictionary"},
                    "index": {"$first": "$index"},
                }
            },
        ]
        return self.attn_head_collection.aggregate(pipeline).next()
        
