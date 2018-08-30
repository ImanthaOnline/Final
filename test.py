from app import app
import unittest


class FlaskTestCase(unittest.TestCase):

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def search(self):
        tester = app.test_client(self)
        response = tester.get('/search/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def find(self):
        tester = app.test_client(self)
        response = tester.get('/find/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def identify(self):
            tester = app.test_client(self)
            response = tester.get('/identify/', content_type='html/text')
            self.assertEqual(response.status_code, 200)

    def view(self):
            tester = app.test_client(self)
            response = tester.get('/view/', content_type='html/text')
            self.assertEqual(response.status_code, 200)


    def criminalinfo(self):
            tester = app.test_client(self)
            response = tester.get('/criminalinfo/', content_type='html/text')
            self.assertEqual(response.status_code, 200)



if __name__ == '__main__':
    unittest.main()
