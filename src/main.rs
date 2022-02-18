use std::{
    f64::consts::PI,
    fmt::Debug,
    fs,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use rand::{prelude::SmallRng, SeedableRng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/* Basics */

#[derive(Debug)]
struct RGB(u8, u8, u8);

///
/// **8-bit color
///
#[derive(Debug)]
struct PPM {
    pub image_width: usize,
    pub image_height: usize,
    pub pixels: Vec<RGB>,
}

impl PPM {
    fn write(&self, path: &'static str) {
        assert_eq!(self.pixels.len(), self.image_width * self.image_height);

        let header = format!("P3\n{} {}\n255\n", self.image_width, self.image_height);
        let mut body = String::new();
        for p in &self.pixels {
            body.push_str(&format!("{} {} {}\n", p.0, p.1, p.2))
        }
        fs::write(path, header + &body).unwrap()
    }
}

///
/// `(double, double, double)`
///
#[derive(Debug, Clone, Copy)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    fn x(&self) -> f64 {
        self.0
    }

    fn y(&self) -> f64 {
        self.1
    }

    #[allow(dead_code)]
    fn z(&self) -> f64 {
        self.2
    }

    fn length(&self) -> f64 {
        f64::sqrt(self.length_squared())
    }

    fn length_squared(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    // fn minus(&self) -> Vec3 {
    //     Vec3(0.0 - self.0, 0.0 - self.1, 0.0 - self.2)
    // }

    fn dot(l: &Vec3, r: &Vec3) -> f64 {
        l.0 * r.0 + l.1 * r.1 + l.2 * r.2
    }

    fn cross(l: &Vec3, r: &Vec3) -> Vec3 {
        Vec3(
            l.1 * r.2 - l.2 * r.1,
            l.2 * r.0 - l.0 * r.2,
            l.0 * r.1 - l.1 * r.0,
        )
    }

    fn unit_vector(v: Vec3) -> Vec3 {
        v / v.length()
    }

    fn random() -> Vec3 {
        Vec3(random_double(), random_double(), random_double())
    }

    fn random_clamp(min: f64, max: f64) -> Vec3 {
        Vec3(
            random_double_clamp(min, max),
            random_double_clamp(min, max),
            random_double_clamp(min, max),
        )
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        f64::abs(self.0) < s && f64::abs(self.1) < s && f64::abs(self.2) < s
    }

    fn reflect(v_in: &Vec3, normal: &Vec3) -> Vec3 {
        *v_in - *normal * Vec3::dot(v_in, normal) * 2.0
    }

    fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = f64::min(Vec3::dot(&-*uv, n), 1.0);
        let r_out_perp = (*uv + *n * cos_theta) * etai_over_etat;
        let r_out_parallel = *n * -f64::sqrt(f64::abs(1.0 - r_out_perp.length_squared()));
        r_out_perp + r_out_parallel
    }

    fn random_in_unit_disk() -> Vec3 {
        loop {
            let p = Vec3(
                random_double_clamp(-1.0, 1.0),
                random_double_clamp(-1.0, 1.0),
                0.0,
            );
            if p.length_squared() >= 1.0 {
                continue;
            }

            return p;
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}

impl Sum for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result = Vec3::origin();
        for i in iter {
            result = result + i;
        }
        result
    }
}

type Point3 = Vec3;

impl Point3 {
    fn origin() -> Point3 {
        Vec3(0.0, 0.0, 0.0)
    }
}

type Color3 = Vec3;

impl Color3 {
    fn white() -> Color3 {
        Vec3(1.0, 1.0, 1.0)
    }

    fn black() -> Color3 {
        Vec3(0.0, 0.0, 0.0)
    }

    fn r(&self) -> f64 {
        self.0
    }

    fn g(&self) -> f64 {
        self.1
    }

    fn b(&self) -> f64 {
        self.2
    }
}

/* Hit Test */

#[derive(Debug)]
struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Point3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    fn at(&self, time: f64) -> Point3 {
        self.origin + self.direction * time
    }
}

#[derive(Debug)]
struct HitRecord {
    p: Point3,
    normal: Vec3,
    t: f64,
    front_face: Option<bool>,
    mat_ptr: MatPtr,
}

impl HitRecord {
    fn new(point: Point3, normal: Vec3, t: f64, mat_ptr: MatPtr) -> HitRecord {
        HitRecord {
            p: point,
            normal,
            t,
            mat_ptr,
            front_face: None,
        }
    }

    fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec3) {
        let is_front = Vec3::dot(&r.direction, outward_normal) < 0.0;
        self.front_face = Some(is_front);
        if is_front {
            self.normal = *outward_normal;
        } else {
            self.normal = -*outward_normal;
        }
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}


struct Sphere {
    center: Point3,
    radius: f64,
    mat_ptr: MatPtr,
}

impl Sphere {
    fn new(center: Point3, radius: f64, mat_ptr: MatPtr) -> Sphere {
        Sphere {
            center,
            radius,
            mat_ptr,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = Vec3::dot(&oc, &r.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminat = half_b * half_b - a * c;

        if discriminat < 0.0 {
            return None;
        }

        let sqrd = f64::sqrt(discriminat);
        let mut root = (-half_b - sqrd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrd) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }

        let point = r.at(root);
        let normal = (point - self.center) / self.radius;
        let mut hr = HitRecord::new(point, normal, root, self.mat_ptr.clone());
        let out_normal = (point - self.center) / self.radius;
        hr.set_face_normal(r, &out_normal);

        Some(hr)
    }
}

struct HittableList(Vec<Arc<dyn Hittable>>);

unsafe impl Sync for HittableList {}

impl HittableList {
    fn new() -> HittableList {
        HittableList(vec![])
    }

    fn add(&mut self, value: Arc<dyn Hittable>) {
        self.0.push(value)
    }

    // fn clear(&mut self) {
    //     self.0.clear()
    // }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut temp_rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;

        for h in &self.0 {
            if let Some(hr) = h.hit(ray, t_min, closest_so_far) {
                closest_so_far = hr.t;
                temp_rec = Some(hr);
            }
        }

        temp_rec
    }
}

#[derive(Debug)]
struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f64,
}

impl Camera {
    ///
    /// `vfov`: vertical field-of-view in degrees
    ///
    fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        apeture: f64,
        focus_dist: f64,
    ) -> Camera {
        let theta = degrees_to_radians(vfov);
        let h = f64::tan(theta / 2.0);
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = Vec3::unit_vector(lookfrom - lookat);
        let u = Vec3::unit_vector(Vec3::cross(&vup, &w));
        let v = Vec3::cross(&w, &u);

        let origin = lookfrom;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        let lens_radius = apeture / 2.0;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = Vec3::random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x() + self.v * rd.y();

        Ray::new(
            self.origin + offset,
            self.lower_left_corner + self.horizontal * s + self.vertical * t - self.origin - offset,
        )
    }
}

struct ScatterOut {
    r_out: Ray,
    attenuation: Color3,
}

/* Material */

trait Material: Debug {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<ScatterOut>;
}

type MatPtr = Arc<dyn Material>;

#[derive(Debug)]
struct Lambertian {
    albedo: Color3,
}

impl Lambertian {
    fn new(color: Color3) -> Lambertian {
        Lambertian { albedo: color }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Option<ScatterOut> {
        let mut scatter_direction = rec.normal + random_unit_vector();

        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        let r_out = Ray::new(rec.p, scatter_direction);
        let attenuation = self.albedo;
        Some(ScatterOut { r_out, attenuation })
    }
}

#[derive(Debug)]
struct Metal {
    albedo: Color3,
    fuzz: f64,
}

impl Metal {
    fn new(color: Color3, fuzz: f64) -> Metal {
        Metal {
            albedo: color,
            fuzz: f64::max(fuzz, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<ScatterOut> {
        let reflected = Vec3::reflect(&Vec3::unit_vector(r_in.direction), &rec.normal);
        let scattered = Ray::new(rec.p, reflected);
        let attenuation = self.albedo;
        if Vec3::dot(&scattered.direction, &rec.normal) > 0.0 {
            Some(ScatterOut {
                r_out: scattered,
                attenuation,
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Dielectric {
    ir: f64, // Index of Refraction
}

impl Dielectric {
    fn new(ir: f64) -> Dielectric {
        Dielectric { ir }
    }

    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        // Use Schlick's approximation for reflectance.
        let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * f64::powi(1.0 - cosine, 5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<ScatterOut> {
        let attenuation = Color3::white();
        let refraction_ratio;
        match rec.front_face {
            Some(true) => refraction_ratio = 1.0 / self.ir,
            _ => refraction_ratio = self.ir,
        }

        let unit_direction = Vec3::unit_vector(r_in.direction);
        let cos_theta = f64::min(Vec3::dot(&-unit_direction, &rec.normal), 1.0);
        let sin_theta = f64::sqrt(1.0 - cos_theta * cos_theta);
        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction;
        if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > random_double()
        {
            direction = Vec3::reflect(&unit_direction, &rec.normal);
        } else {
            direction = Vec3::refract(&unit_direction, &rec.normal, refraction_ratio);
        }

        Some(ScatterOut {
            attenuation,
            r_out: Ray::new(rec.p, direction),
        })
    }
}

/* Utility fn */

fn c2c(c: Color3) -> RGB {
    RGB(f2u(c.0), f2u(c.1), f2u(c.2))
}

fn c2c_downsample(c: Color3, samples_per_pixel: i64) -> RGB {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = f64::sqrt(c.r() * scale);
    let g = f64::sqrt(c.g() * scale);
    let b = f64::sqrt(c.b() * scale);
    c2c(Vec3(r, g, b))
}

fn f2u(f: f64) -> u8 {
    if f > 1.0 {
        255
    } else if f < 0.0 {
        0
    } else {
        (255.0 * f) as u8
    }
}

///
/// core func
///
fn ray_color(r: &Ray, world: &dyn Hittable, depth: i64) -> Color3 {
    if depth <= 0 {
        // block by others
        return Color3::black();
    }

    if let Some(rec) = world.hit(r, 0.001, f64::INFINITY) {
        // go recur, till we found some light
        match rec.mat_ptr.scatter(r, &rec) {
            Some(ScatterOut { r_out, attenuation }) => {
                return ray_color(&r_out, world, depth - 1) * attenuation
            }
            None => return Color3::black(),
        }
    }

    // some kinda *global-illumination*
    // source of all colors
    let unit_direction = Vec3::unit_vector(r.direction);
    let t = 0.5 * (unit_direction.y() + 1.0);

    // gradient blue sky box at y-axis
    // by lerp
    Color3::white() * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t
}

fn degrees_to_radians(deg: f64) -> f64 {
    deg * PI / 180.0
}

///
/// has to be effient & thread-safe
/// 
fn random_double() -> f64 {
    // random()
    let mut rng = SmallRng::from_entropy();
    rng.gen()
}

fn random_double_clamp(min: f64, max: f64) -> f64 {
    min + (max - min) * random_double()
}

// fn clamp(x: f64, min: f64, max: f64) -> f64 {
//     if x < min {
//         return min;
//     } else if x > max {
//         return max;
//     } else {
//         return x;
//     }
// }

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::random_clamp(-1.0, 1.0);
        if p.length_squared() >= 1.0 {
            continue;
        }

        return p;
    }
}

fn random_unit_vector() -> Vec3 {
    Vec3::unit_vector(random_in_unit_sphere())
}

///
/// section 2.1
///
#[allow(dead_code)]
fn write_ppm_sample() {
    let image_width: usize = 3;
    let image_height: usize = 2;
    let pixels = vec![
        RGB(255, 0, 0),
        RGB(0, 255, 0),
        RGB(0, 0, 255),
        RGB(255, 255, 0),
        RGB(255, 255, 255),
    ];
    let ppm = PPM {
        image_width,
        image_height,
        pixels,
    };
    ppm.write("foo.ppm")
}

fn random_scene() -> HittableList {
    let mut world = HittableList::new();

    let ground_material = Arc::new(Lambertian::new(Vec3(0.5, 0.5, 0.5)));
    world.add(Arc::new(Sphere::new(
        Vec3(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    )));

    // random little balls
    for x in -11..11 {
        for z in -11..11 {
            let choose_mat = random_double();
            let center = Vec3(
                x as f64 + 0.9 * random_double(),
                0.2,
                z as f64 + 0.9 * random_double(),
            );

            if (center - Vec3(4.0, 0.2, 0.0)).length() > 0.9 {
                let sphere_material: MatPtr;

                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Vec3::random() * Vec3::random();
                    sphere_material = Arc::new(Lambertian::new(albedo));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Vec3::random_clamp(0.5, 1.0);
                    let fuzz = random_double_clamp(0.0, 0.5);
                    sphere_material = Arc::new(Metal::new(albedo, fuzz));
                } else {
                    // glass
                    sphere_material = Arc::new(Dielectric::new(1.5));
                }

                world.add(Arc::new(Sphere::new(center, 0.2, sphere_material)));
            }
        }
    }

    // three big ones
    let glass = Arc::new(Dielectric::new(1.5));
    world.add(Arc::new(Sphere::new(Vec3(0.0, 1.0, 0.0), 1.0, glass)));
    let diffuse = Arc::new(Lambertian::new(Vec3(0.4, 0.2, 0.1)));
    world.add(Arc::new(Sphere::new(Vec3(-4.0, 1.0, 0.0), 1.0, diffuse)));
    let metal = Arc::new(Metal::new(Vec3(0.7, 0.6, 0.5), 0.0));
    world.add(Arc::new(Sphere::new(Vec3(4.0, 1.0, 0.0), 1.0, metal)));

    world
}

fn run_the_world() {
    // Image (screen)

    let aspect_ratio = 3.0 / 2.0;
    let image_width = 600.0;
    let image_height = image_width / aspect_ratio;
    let samples_per_pixel = 40; // 40x anti-aliasing
    let max_depth = 50;

    // World

    let world = random_scene();

    // Camera

    let lookfrom = Vec3(13.0, 2.0, 3.0);
    let lookat = Vec3::origin();
    let vup = Vec3(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;

    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    // Render

    let mut j = image_height as i64 - 1;
    let mut i: i64;
    let mut pixels = vec![];

    while j >= 0 {
        i = 0;
        while i < image_width as i64 {
            if samples_per_pixel > 0 {
                // anti-aliasing
                // multi-threaded
                let c: Color3 = (0..samples_per_pixel)
                    .into_par_iter()
                    .map(|_| {
                        let u = (i as f64 + random_double()) / (image_width - 1.0);
                        let v = (j as f64 + random_double()) / (image_height - 1.0);
                        let ray = camera.get_ray(u, v);
                        ray_color(&ray, &world, max_depth)
                    })
                    .sum();
                pixels.push(c2c_downsample(c, samples_per_pixel));
            } else {
                let u = i as f64 / (image_width - 1.0);
                let v = j as f64 / (image_height - 1.0);
                let ray = camera.get_ray(u, v);
                let c = ray_color(&ray, &world, max_depth);
                pixels.push(c2c(c));
            }

            i += 1;
        }

        j -= 1;
    }

    let ppm = PPM {
        pixels,
        image_height: image_height as usize,
        image_width: image_width as usize,
    };
    ppm.write("final.ppm")
}

fn main() {
    // write_ppm_sample()
    run_the_world()
}
