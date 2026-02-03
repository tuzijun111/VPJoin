


---

## 🚀 Instructions

To generate proofs for SQL queries **Q1, Q3, Q5, Q8, Q9, and Q18**, run the following commands from the project root:

```bash

# Query 3
cargo test --package halo2-experiments --lib -- sql::q3_final_v7::tests::test_1 --exact --nocapture

# Query 5
cargo test --package halo2-experiments --lib -- sql::q5_final_v4::tests::test_1 --exact --nocapture

# Query 8
cargo test --package halo2-experiments --lib -- sql::q8_final_v3::tests::test_1 --exact --nocapture

# Query 9
cargo test --package halo2-experiments --lib -- sql::q9_final_v2::tests::test_1 --exact --nocapture

# Query 18
cargo test --package halo2-experiments --lib -- sql::q18_final_v2::tests::test_1 --exact --nocapture
```

---

## 📝 Notes

### **Prerequisite: Stack Size**

Halo2 circuits require a sufficiently large stack size. Set `RUST_MIN_STACK` before generating proofs:

```bash
export RUST_MIN_STACK=33554432
```

---

### **Public Parameter Selection (`k`)**

Select appropriate Halo2 public parameters depending on dataset size:

| Dataset Size | Queries (Q3, Q5, Q8, Q9, Q18) 
| ------------ | --------------------------------- 
| 60k Rows     | k = 16                        
| 120k Rows    | k = 17                        
| 240k Rows    | k = 18                        

---






