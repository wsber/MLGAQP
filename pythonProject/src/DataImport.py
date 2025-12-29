from neo4j import GraphDatabase
import csv
from tqdm import tqdm  # 进度条库

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4jws"))

query = """
MATCH (u:User)-[:view]->(p:Post)-[:have]->(c:Comment)<-[:public]-(u)
RETURN 
    id(u) AS userId, 
    id(p) AS postId, 
    id(c) AS commentId,
    p.ML1_oracle1_probability AS post_oracle1,
    p.ML1_proxy4b1_probability AS post_proxy4b1,
    p.ML1_proxy2b1_probability AS post_proxy2b1,
    c.ML2_oracle2_probability AS comment_oracle2,
    c.ML2_proxy2d2_probability AS comment_proxy2d2,
    c.ML2_proxy4d2_probability AS comment_proxy4d2,
    c.upvotes AS comment_upvotes
"""
data_dir = '/home/wangshuo/projects/Neo4j_Exp/pythonProject/output/'
with driver.session() as session:
    result = session.run(query)

    with open(data_dir + "subgraph_ids.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "userId", "postId", "commentId",
            "post_oracle1", "post_proxy4b1", "post_proxy2b1",
            "comment_oracle2", "comment_proxy2d2", "comment_proxy4d2", "comment_upvotes"
        ])
        for record in tqdm(result, desc="导出进度", unit="条"):
            writer.writerow([
                record["userId"], record["postId"], record["commentId"],
                record["post_oracle1"], record["post_proxy4b1"], record["post_proxy2b1"],
                record["comment_oracle2"], record["comment_proxy2d2"], record["comment_proxy4d2"], record["comment_upvotes"]
            ])
